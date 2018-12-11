#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <std_msgs/String.h>
#include <std_msgs/Float64.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/ChannelFloat32.h>

#include <sstream>
#include <mutex>
#include <time.h>

#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <queue>

#include <upsample/CollisionAvoidance.h>

// bebop 1
// #define Ku 338.7884
// #define Kv 444.1741
// #define U0 256.0
// #define V0 192.0

// #define ORB_WIDTH 640.f
// #define ORB_HEIGHT 368.f

// bebop 2
#define Ku 566.411826
#define Kv 555.217642
#define U0 408.067799
#define V0 237.170367

#define ORB_WIDTH 856.f
#define ORB_HEIGHT 480.f

#define MEGA_WIDTH 512.f
#define MEGA_HEIGHT 384.f

#define SQUARED_SIGMA_S 100.f
#define SIGMA 0.5f
#define SQUARED_SIGMA_R (SIGMA * SIGMA)

#define FRAME_ID_WORLD "odom"
#define FRAME_ID_CAMERA "base_link"

#define PATCH_HALF_SIZE 5
#define OG_SIZE 0.1
#define OG_CLIP 5
#define OG_CLIP_WORLD 10

#define DO_BLUR false
#define VISUALIZE_MEGA_PCL true
#define VISUALIZE_ORB_PCL true
#define VISUALIZE_US_PCL true
#define VISUALIZE_US_OG true
#define VISUALIZE_US_GOAL true

#define REPULSIVE_FORCE 0.02
#define ATTRACTIVE_FORCE 1
#define PROGRESSIVE_RATIO 0.2

// std::mutex pcl_mutex;

using namespace std;

class UpSampler
{
private:
	mutex orb_lock;
	mutex pose_lock;

	mutex world_og_lock;

public:
	// ros::NodeHandle nh_;
	// image_transport::ImageTransport it_;
	ros::ServiceServer service_;
	ros::Subscriber mega_sub_, orb_sub_, pose_sub_, scale_sub_, fix_og_sub_;
	ros::Publisher mega_pcl_pub_, orb_pcl_pub_, us_pcl_pub_, us_og_pub_, tri_img_pub_, scale_img_pub_, us_img_pub_;
	ros::Publisher current_position_pub_, goal_position_pub_, intermediate_position_pub_;
	ros::Publisher us_world_og_pub_;

	// to sync pcl_cam with pose
	queue<sensor_msgs::PointCloudConstPtr > q_orbmsg;
	queue<geometry_msgs::PoseStampedConstPtr > q_posemsg;

	// local OG
	sensor_msgs::PointCloud ogmsg;	// local occupancy grid
	geometry_msgs::PoseStampedConstPtr posemsg;	// local occupancy grid' pose

	// global OG
	bool fix_world_og;
	sensor_msgs::PointCloud world_ogmsg;
	map<int, float> world_og_score;

	int NUM_WORLD_Z_GRIDS, NUM_WORLD_Y_GRIDS;

	void reset_world_og()
	{

		world_ogmsg.points.clear();
		world_ogmsg.channels.clear();

		world_og_lock.lock();
		fix_world_og = false;
		world_og_score.clear();
		world_og_lock.unlock();
	}

	int hash_world_og(float x, float y, float z)
	{
		if (x <= -OG_CLIP_WORLD || x >= OG_CLIP_WORLD || 
			y <= -OG_CLIP_WORLD || y >= OG_CLIP_WORLD || 
			z <= -OG_CLIP_WORLD || z >= OG_CLIP_WORLD)
			return -1;
		return floor((x + OG_CLIP_WORLD) / OG_SIZE) * NUM_WORLD_Y_GRIDS * NUM_WORLD_Z_GRIDS + 
				floor((y + OG_CLIP_WORLD) / OG_SIZE) * NUM_WORLD_Z_GRIDS + floor((z + OG_CLIP_WORLD) / OG_SIZE);
	} 

	float mfscale;
	// std::lock_guard<std::mutex> lock(pcl_mutex)

	UpSampler()
	{
		mfscale = 1.f;
		fix_world_og = false;

		NUM_WORLD_Z_GRIDS = ceil(2 * OG_CLIP_WORLD / OG_SIZE) + 1;
 		NUM_WORLD_Y_GRIDS = NUM_WORLD_Z_GRIDS;
	}

	~UpSampler()
	{
	}

	void imageCb(const sensor_msgs::ImageConstPtr& msg)
	{
		ROS_INFO("inv_depth_image received");
		clock_t timer = clock();
		clock_t timer_total = clock();
		orb_lock.lock();
		pose_lock.lock();

		if (q_orbmsg.size() == 0 || q_posemsg.size() == 0)
		{
			orb_lock.unlock();
			pose_lock.unlock();
			return;
		}
		// sync with pcl_cam
		sensor_msgs::PointCloudConstPtr orbmsg = q_orbmsg.front();
		while (q_orbmsg.size() > 0 && q_orbmsg.front()->header.stamp.toSec() < msg->header.stamp.toSec())
		{
			orbmsg = q_orbmsg.front();
			q_orbmsg.pop();
		}
		if (q_orbmsg.size() > 0 && q_orbmsg.front()->header.stamp.toSec() - msg->header.stamp.toSec() < msg->header.stamp.toSec() - orbmsg->header.stamp.toSec())
		{
			orbmsg = q_orbmsg.front();
			q_orbmsg.pop();
		}

		// sync with unscaled_pose
		// geometry_msgs::PoseStampedConstPtr 
		posemsg = q_posemsg.front();
		while (q_posemsg.size() > 0 && q_posemsg.front()->header.stamp.toSec() < msg->header.stamp.toSec())
		{
			posemsg = q_posemsg.front();
			q_posemsg.pop();
		}
		if (q_posemsg.size() > 0 && q_posemsg.front()->header.stamp.toSec() - msg->header.stamp.toSec() < msg->header.stamp.toSec() - posemsg->header.stamp.toSec())
		{
			posemsg = q_posemsg.front();
			q_posemsg.pop();
		}

		ROS_INFO("<%u> time used to sync %f", msg->header.seq, float(clock() - timer) / CLOCKS_PER_SEC);
		orb_lock.unlock();
		pose_lock.unlock();

		cv_bridge::CvImagePtr cv_ptr, cv_ptr_copy_tri, cv_ptr_copy_scale, cv_ptr_copy_us;
		try
		{
			cv_ptr = cv_bridge::toCvCopy(msg, "32FC1");
			cv_ptr_copy_tri = cv_bridge::toCvCopy(msg, "32FC1");
			cv_ptr_copy_scale = cv_bridge::toCvCopy(msg, "32FC1");
			cv_ptr_copy_us = cv_bridge::toCvCopy(msg, "32FC1");
		}
		catch(cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}

		if (VISUALIZE_MEGA_PCL)
		{
			sensor_msgs::PointCloud pclMsg;

			for (size_t v = 0; v < cv_ptr->image.rows; v++)
				for (size_t u = 0; u < cv_ptr->image.cols; u++)
				{
					float d = exp(1.f / cv_ptr->image.at<float>(v,u));

					float xc = (u - U0) / Ku;
					float yc = (v - V0) / Kv;

					float zz = d / sqrt(xc * xc + yc * yc + 1);	// x and y depend on z
					float xx = xc * zz;
					float yy = yc * zz;
	
					geometry_msgs::Point32 pt;
					// apply_Rt_to_pt(xx, yy, zz, posemsg->pose.position.x, posemsg->pose.position.y, posemsg->pose.position.z, 
					// 	 posemsg->pose.orientation.w, posemsg->pose.orientation.x, posemsg->pose.orientation.y, posemsg->pose.orientation.z, 
					// 	pt.x, pt.y, pt.z);
					// pt.x *= mfscale;
					// pt.y *= mfscale;
					// pt.z *= mfscale;
					pt.x = xx;
					pt.y = yy;
					pt.z = zz;
					pclMsg.points.push_back(pt);
				}
			pclMsg.header = cv_ptr->header;
			pclMsg.header.frame_id = FRAME_ID_CAMERA;
			pclMsg.header.stamp = ros::Time::now();
	
			mega_pcl_pub_.publish(pclMsg);
		}
		

		// 
		// repub orb
		// 
		

		if(orbmsg == NULL || posemsg == NULL)
			return;
		int num_points = orbmsg->points.size();
		ROS_INFO("<%u> processing orb message---, %d", msg->header.seq, num_points);
		if (VISUALIZE_ORB_PCL)
		{
			orb_pcl_pub_.publish(orbmsg);
		}

		// 
		// upsampling using triangluate approach
		// 

		// STEP 1: remove outlier

		// perform the delaunary 
		
		timer = clock();
		cv::Rect rect(0,0, int(MEGA_WIDTH), int(MEGA_HEIGHT));
		cv::Subdiv2D subdiv_raw(rect);

		// prepare for orb_depth look up to remove duplicate point after scaling
		cv::Mat mat_orb_depth(MEGA_HEIGHT, MEGA_WIDTH, CV_32FC1, -1); // default depth is set to be -1
		
		int num_unique_points = num_points + 4;	// including the virual points used by Subdiv2D
		int max_pt_id = 0;
		vector<float> vec_log_scale_raw(num_unique_points);

		for (int i = 0; i < num_points; i++)
		{
			// use the u, v channels
			float tmp_u = (orbmsg->channels[0].values[i] / ORB_WIDTH * MEGA_WIDTH);
			float tmp_v = (orbmsg->channels[1].values[i] / ORB_HEIGHT * MEGA_HEIGHT);
			float tmp_orb_depth = sqrt(orbmsg->points[i].x * orbmsg->points[i].x + orbmsg->points[i].y * orbmsg->points[i].y + orbmsg->points[i].z * orbmsg->points[i].z);
			cv::Point2f tmp_pt(tmp_u, tmp_v);

			// use the pt_id to refer a patch
			int pt_id = subdiv_raw.insert(tmp_pt);
			// existing pt_id
			if (pt_id <= max_pt_id)
			{
				num_unique_points --;
				vec_log_scale_raw.resize(num_unique_points);

				// existing depth is closer than the new one, do nothing
				if (mat_orb_depth.at<float>(cvRound(tmp_v), cvRound(tmp_u)) <= tmp_orb_depth)
					continue;
			}
			else{
				max_pt_id = pt_id;
			}

			mat_orb_depth.at<float>(cvRound(tmp_v), cvRound(tmp_u)) = tmp_orb_depth;
			float tmp_log_scale = log(tmp_orb_depth) - 1.f / look_up_pt_inv_log_mega_depth(tmp_pt, cv_ptr->image);
			vec_log_scale_raw[pt_id] = tmp_log_scale;
			// ROS_INFO("<%u> pt_id %d : log_scale %f", msg->header.seq, pt_id, tmp_log_scale);
			// ROS_INFO("pt %d with pt_id  %d at (%f, %f)", i, pt_id, tmp_u, tmp_v);
		}	// note taht pt_id 0,1,2,3 are not handled

		// neighbours decide outliers
		set<int> set_outlier_pt_id;
		vector<set<int> > vec_set_neighbour(num_unique_points);
		vector<set<int> > vec_set_neighbour_2(num_unique_points);

		for (int pt_id = 4; pt_id < num_unique_points; pt_id++)
		{
			pt_neighbours_in_subdiv(pt_id, subdiv_raw, vec_set_neighbour[pt_id], true);
		}
		for (int pt_id = 4; pt_id < num_unique_points; pt_id++)
		{
			for (set<int>::iterator it = vec_set_neighbour[pt_id].begin(); it != vec_set_neighbour[pt_id].end(); it++)
			{
				union_of_sets(vec_set_neighbour[*it], vec_set_neighbour_2[pt_id]);
			}

			vector<float> vec_neighbour_scale;
			for (set<int>::iterator it = vec_set_neighbour_2[pt_id].begin(); it != vec_set_neighbour_2[pt_id].end(); it++)
			{
				vec_neighbour_scale.push_back(exp(vec_log_scale_raw[*it]));
			}
			if (is_outlier(exp(vec_log_scale_raw[pt_id]), vec_neighbour_scale))
			{
				set_outlier_pt_id.insert(pt_id);
			}
		}

		cv::Subdiv2D subdiv_clean(rect);

		int num_unique_points_clean = num_unique_points - set_outlier_pt_id.size();	// excluding the outliers
		vector<float> vec_log_scale_clean(num_unique_points_clean);
		float scale_average = 0.f;

		for (int pt_id_raw = 4; pt_id_raw < num_unique_points; pt_id_raw++)
		{	
			// pt_id_raw is an outlier
			if (set_outlier_pt_id.find(pt_id_raw) != set_outlier_pt_id.end())
				continue;

			int pt_id = subdiv_clean.insert(subdiv_raw.getVertex(pt_id_raw));
			vec_log_scale_clean[pt_id] = vec_log_scale_raw[pt_id_raw];
		}
		// average scale for pt_id 0-3
		float log_scale_average = accumulate(vec_log_scale_raw.begin() + 4, vec_log_scale_raw.end(), 0.f) / (num_unique_points - 4);
		for (int pt_id = 0; pt_id < 4; pt_id++)
		{
			vec_log_scale_clean[pt_id] = log_scale_average;
		}

		ROS_INFO("<%u> time used to remove outliers %f", msg->header.seq, float(clock() - timer) / CLOCKS_PER_SEC);
		// step 2.2: compute the scale
		
		timer = clock();
		float time_used_to_locate = 0.f;
		float time_used_to_prepare = 0.f;
		float time_used_to_compute = 0.f;
		//clock_t timer;

		float scale_max = -1.f, scale_min = 9999.f;
		float scale_sum = 0.f;
		float scale_count = 0.f;
		cv::Mat mat_log_scale(MEGA_HEIGHT, MEGA_WIDTH, CV_32FC1, -1); // default scale is set to be -1
// #pragma omp parallel
// #pragma omp for	
		for (size_t v = 0; v < cv_ptr->image.rows; v++)
		{
			for (size_t u = 0; u < cv_ptr->image.cols; u++)
			{
				cv::Point2f pt(u, v);
				float log_mega_depth = 1.f / look_up_pt_inv_log_mega_depth(pt, cv_ptr->image);

				// timer = clock();
				int v1,v2,v3,v4,v5,v6;
				int flag = locate_pt_in_subdiv_opt(pt, subdiv_clean, v1,v2,v3,v4,v5,v6);
				// timer = clock() - timer;
				// time_used_to_locate += ((float)timer)/CLOCKS_PER_SEC;

				// timer = clock();
				switch (flag)
				{
					case cv::Subdiv2D::PTLOC_INSIDE:
					{
						float scale1 = exp(vec_log_scale_clean[v1]);
						float scale2 = exp(vec_log_scale_clean[v2]);
						float scale3 = exp(vec_log_scale_clean[v3]);
						float scale4 = exp(vec_log_scale_clean[v4]);
						float scale5 = exp(vec_log_scale_clean[v5]);
						float scale6 = exp(vec_log_scale_clean[v6]);

						float diff1 = squared_diff_in_log_mega_depth_from(log_mega_depth, subdiv_clean.getVertex(v1), cv_ptr->image);
						float diff2 = squared_diff_in_log_mega_depth_from(log_mega_depth, subdiv_clean.getVertex(v2), cv_ptr->image);
						float diff3 = squared_diff_in_log_mega_depth_from(log_mega_depth, subdiv_clean.getVertex(v3), cv_ptr->image);
						float diff4 = squared_diff_in_log_mega_depth_from(log_mega_depth, subdiv_clean.getVertex(v4), cv_ptr->image);
						float diff5 = squared_diff_in_log_mega_depth_from(log_mega_depth, subdiv_clean.getVertex(v5), cv_ptr->image);
						float diff6 = squared_diff_in_log_mega_depth_from(log_mega_depth, subdiv_clean.getVertex(v6), cv_ptr->image);

						float weight1 = exp(- diff1 / SQUARED_SIGMA_R);
						float weight2 = exp(- diff2 / SQUARED_SIGMA_R);
						float weight3 = exp(- diff3 / SQUARED_SIGMA_R);
						float weight4 = exp(- diff4 / SQUARED_SIGMA_R);
						float weight5 = exp(- diff5 / SQUARED_SIGMA_R);
						float weight6 = exp(- diff6 / SQUARED_SIGMA_R);

						float weighted_scale_sum = weight1 * scale1 + weight2 * scale2 + weight3 * scale3 + weight4 * scale4 + weight5 * scale5 + weight6 * scale6;
						float weight_sum = weight1 + weight2 + weight3 + weight4 + weight5 + weight6;
						mat_log_scale.at<float>(v, u) = log(weighted_scale_sum / weight_sum);
						break;
					}
					case cv::Subdiv2D::PTLOC_ON_EDGE: 
					{
						float scale1 = exp(vec_log_scale_clean[v1]);
						float scale2 = exp(vec_log_scale_clean[v2]);
						float scale3 = exp(vec_log_scale_clean[v3]);
						float scale4 = exp(vec_log_scale_clean[v4]);

						float diff1 = squared_diff_in_log_mega_depth_from(log_mega_depth, subdiv_clean.getVertex(v1), cv_ptr->image);
						float diff2 = squared_diff_in_log_mega_depth_from(log_mega_depth, subdiv_clean.getVertex(v2), cv_ptr->image);
						float diff3 = squared_diff_in_log_mega_depth_from(log_mega_depth, subdiv_clean.getVertex(v3), cv_ptr->image);
						float diff4 = squared_diff_in_log_mega_depth_from(log_mega_depth, subdiv_clean.getVertex(v4), cv_ptr->image);

						float weight1 = exp(- diff1 / SQUARED_SIGMA_R);
						float weight2 = exp(- diff2 / SQUARED_SIGMA_R);
						float weight3 = exp(- diff3 / SQUARED_SIGMA_R);
						float weight4 = exp(- diff4 / SQUARED_SIGMA_R);

						float weighted_scale_sum = weight1 * scale1 + weight2 * scale2 + weight3 * scale3 + weight4 * scale4;
						float weight_sum = weight1 + weight2 + weight3 + weight4;
						mat_log_scale.at<float>(v, u) = log(weighted_scale_sum / weight_sum);
						break;
					}
					case cv::Subdiv2D::PTLOC_VERTEX:
					{ 
						mat_log_scale.at<float>(v, u) = vec_log_scale_clean[v1];
						break;
					}
				}
				// timer = clock() - timer;
				// time_used_to_compute += ((float)timer)/CLOCKS_PER_SEC;
			}
		}

		// ROS_INFO("<%u> time to locate: %f, time to prepare: %f, time to compute: %f", msg->header.seq, time_used_to_locate, time_used_to_prepare, time_used_to_compute);
		// if (DO_BLUR)
		// 	cv::GaussianBlur(mat_scale, mat_scale, cv::Size(0,0), 10);

		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(mat_log_scale, &minVal, &maxVal, &minLoc, &maxLoc);
		// ROS_INFO("<%u> max scale is %lf, min scale is %lf",  msg->header.seq, exp(maxVal), exp(minVal));
		ROS_INFO("<%u> time to compute the scales %f", msg->header.seq, float(clock() - timer)/CLOCKS_PER_SEC);

		// visualize triangulation
		vector<cv::Point3f> points_to_draw, neighbour_points_to_draw;
		float max_log_scale_clean = max_in_vector(vec_log_scale_clean);
		for (int i = 0; i < num_unique_points_clean; i++)
		{
			cv::Point2f tmp_pt = subdiv_clean.getVertex(i);
			points_to_draw.push_back(cv::Point3f(tmp_pt.x, tmp_pt.y, vec_log_scale_clean[i] / max_log_scale_clean));
		}
		// int v1,v2,v3,v4,v5,v6;
		// locate_pt_in_subdiv_opt(cv::Point2f(100.f, 100.f), subdiv_clean,v1,v2,v3,v4,v5,v6);
		// neighbour_points_to_draw.push_back(cv::Point3f(100.f, 100.f, 10.f));
		// neighbour_points_to_draw.push_back(cv::Point3f(subdiv_clean.getVertex(v1).x, subdiv_clean.getVertex(v1).y, 10.f));
		// neighbour_points_to_draw.push_back(cv::Point3f(subdiv_clean.getVertex(v2).x, subdiv_clean.getVertex(v2).y, 10.f));
		// neighbour_points_to_draw.push_back(cv::Point3f(subdiv_clean.getVertex(v3).x, subdiv_clean.getVertex(v3).y, 10.f));
		draw_delaunay(cv_ptr_copy_tri->image, subdiv_clean);
		draw_points(cv_ptr_copy_tri->image, points_to_draw);
		// draw_points(cv_ptr_copy_tri->image, neighbour_points_to_draw, true);
		tri_img_pub_.publish(cv_ptr_copy_tri->toImageMsg());

		// visualize the scales
		for (size_t v = 0; v < cv_ptr_copy_scale->image.rows; v++)
			for (size_t u = 0; u < cv_ptr_copy_scale->image.cols; u++)
			{
				cv_ptr_copy_scale->image.at<float>(v,u) = mat_log_scale.at<float>(v,u) / maxVal;
			}
		scale_img_pub_.publish(cv_ptr_copy_scale->toImageMsg());

		// visualize the upsampled output in 2D
		float max_log_scale = -1.f;
		for (size_t v = 0; v < cv_ptr_copy_us->image.rows; v++)
			for (size_t u = 0; u < cv_ptr_copy_us->image.cols; u++)
			{
				cv_ptr_copy_us->image.at<float>(v,u) = mat_log_scale.at<float>(v,u) + (1.f/ cv_ptr_copy_us->image.at<float>(v,u));
				if(max_log_scale < cv_ptr_copy_us->image.at<float>(v,u))
					max_log_scale = cv_ptr_copy_us->image.at<float>(v,u);
			}
		cv_ptr_copy_us->image /= max_log_scale;
		us_img_pub_.publish(cv_ptr_copy_us->toImageMsg());
		// ROS_INFO("max log scale is %f", max_log_scale);

		// visualize the upsampled output in 3
		sensor_msgs::PointCloud pclMsg2;
		// to hash the grid
		set<int> set_og_id;
		int num_z_grids = ceil(OG_CLIP / OG_SIZE) + 1;
		int num_y_grids = 2 * num_z_grids + 1;
		for (size_t v = 0; v < cv_ptr->image.rows; v++)
			for (size_t u = 0; u < cv_ptr->image.cols; u++)
			{
				float scale = exp(mat_log_scale.at<float>(v,u));
				float scaled_d = exp(mat_log_scale.at<float>(v,u) + 1.f / cv_ptr->image.at<float>(v,u));
				float xc = (u - U0) / Ku;
				float yc = (v - V0) / Kv;
				float zz = scaled_d / sqrt(xc * xc + yc * yc + 1);	// x and y depend on z
				float xx = xc * zz;
				float yy = yc * zz;
				
				if (VISUALIZE_US_PCL)
				{
					geometry_msgs::Point32 pt;
					// apply_Rt_to_pt(xx, yy, zz, posemsg->pose.position.x, posemsg->pose.position.y, posemsg->pose.position.z, 
					// 	 posemsg->pose.orientation.w, posemsg->pose.orientation.x, posemsg->pose.orientation.y, posemsg->pose.orientation.z,
					// 	pt.x, pt.y, pt.z);
	
					// pt.x *= mfscale;
					// pt.y *= mfscale;
					// pt.z *= mfscale;
	
					pt.x = xx;
					pt.y = yy;
					pt.z = zz;
					pclMsg2.points.push_back(pt);
				}
				
				// for occupancy grid
				if (xx <= -OG_CLIP || xx >= OG_CLIP || yy <= -OG_CLIP || yy >= OG_CLIP || zz >= OG_CLIP)
					continue;
				int og_id = floor((xx + OG_CLIP) / OG_SIZE) * num_y_grids * num_z_grids + floor((yy + OG_CLIP) / OG_SIZE) * num_z_grids + floor(zz / OG_SIZE);
				set_og_id.insert(og_id);
			}

		if (VISUALIZE_US_PCL)
		{
			pclMsg2.header = cv_ptr->header;
			pclMsg2.header.frame_id = FRAME_ID_CAMERA;
			pclMsg2.header.stamp = ros::Time::now();
			us_pcl_pub_.publish(pclMsg2);
		}

		// for occupancy grid
		ROS_INFO("set_og_id size %ld", set_og_id.size());
		ogmsg.points.clear();

		world_og_lock.lock();
		if (!fix_world_og)
		{
			world_ogmsg.points.clear();
			world_ogmsg.channels.clear();
			world_ogmsg.channels.resize(1);
			world_ogmsg.channels[0].name = "occupancy score";
		}
		world_og_lock.unlock();
		

		// OG coordinate (reference coordinate)
		cv::Mat mat_reference(4, 4, CV_32FC1);
		posemsg_to_mat(posemsg->pose, mat_reference);

		cv::Mat local_grid_point(4, 1, CV_32FC1);
		local_grid_point.at<float>(3,0) = 1;

		cv::Mat world_grid_point(4, 1, CV_32FC1);

		set<int> set_observed_world_og_id;

		world_og_lock.lock();
		if (!fix_world_og)
		{
			for(set<int>::iterator it = set_og_id.begin(); it != set_og_id.end(); it++)
			{
				geometry_msgs::Point32 grid;
				int xxx = (*it) / (num_y_grids * num_z_grids);
				int yyy = ((*it) - xxx * num_y_grids * num_z_grids) / num_z_grids;
				int zzz = (*it) % num_z_grids;
				grid.x = xxx * OG_SIZE - OG_CLIP + 0.5 * OG_SIZE;
				grid.y = yyy * OG_SIZE - OG_CLIP + 0.5 * OG_SIZE;
				grid.z = zzz * OG_SIZE + 0.5 * OG_SIZE;
				ogmsg.points.push_back(grid);
	
				// merge to the world og
				// step 1: transfrom
				local_grid_point.at<float>(0,0) = grid.x;
				local_grid_point.at<float>(1,0) = grid.y;
				local_grid_point.at<float>(2,0) = grid.z;
				world_grid_point = mat_reference * local_grid_point;
	
				// step 2: hash
				int world_og_id = hash_world_og(world_grid_point.at<float>(0,0), world_grid_point.at<float>(1,0), world_grid_point.at<float>(2,0));
				if (world_og_id >= 0)
				{
					set_observed_world_og_id.insert(world_og_id);
	
					// observe an exsiting grid
					if (world_og_score.find(world_og_id) != world_og_score.end())
					{
						world_og_score[world_og_id] += 0.4;
						if (world_og_score[world_og_id] > 1)
							world_og_score[world_og_id] = fabs((world_og_score[world_og_id] - 1.f) / 2.f) + 1.01f;
					} 
					// newly observed grid
					else {
						world_og_score[world_og_id] = 0.4;
					}
				}
			}
	
			for (map<int,float>::iterator it = world_og_score.begin(); it != world_og_score.end(); it++)
			{	
				// unobsered existing grid
				if (set_observed_world_og_id.find(it->first) == set_observed_world_og_id.end())
				{
					if (it->second > 1){
						it->second = fabs((it->second - 1.f) / 2.f) + 1.01f;
					}
					else{
						it->second = it->second / 2.f;
					}
				}
	
				// publish all observed grid
				if (it->second > 0.01)
				{
					geometry_msgs::Point32 world_grid;
					int xxx = it->first / (NUM_WORLD_Y_GRIDS * NUM_WORLD_Z_GRIDS);
					int yyy = (it->first - xxx * NUM_WORLD_Y_GRIDS * NUM_WORLD_Z_GRIDS) / NUM_WORLD_Z_GRIDS;
					int zzz = it->first % NUM_WORLD_Z_GRIDS;
					world_grid.x = xxx * OG_SIZE - OG_CLIP_WORLD + 0.5 * OG_SIZE;
					world_grid.y = yyy * OG_SIZE - OG_CLIP_WORLD + 0.5 * OG_SIZE;
					world_grid.z = zzz * OG_SIZE - OG_CLIP_WORLD + 0.5 * OG_SIZE;
					world_ogmsg.points.push_back(world_grid);
					world_ogmsg.channels[0].values.push_back(it->second);
				}
			}
		}
		world_og_lock.unlock();

		// // insert 5 virtual walls
		// geometry_msgs::Point32 left, right, up, down, back;
		// for(int i = -10; i <= 10; i++)
		// 	for (int j = -10; j <= 10; j++)
		// 	{
		// 		left.x = -1;
		// 		left.y = i * OG_SIZE;
		// 		left.z = j * OG_SIZE;
		// 		ogmsg.points.push_back(left);

		// 		right.x = 1;
		// 		right.y = i * OG_SIZE;
		// 		right.z = j * OG_SIZE;
		// 		ogmsg.points.push_back(right);

		// 		// up.x = i * OG_SIZE;
		// 		// up.y = -1;
		// 		// up.z = j * OG_SIZE;
		// 		// ogmsg.points.push_back(up);


		// 		// down.x = i * OG_SIZE;
		// 		// down.y = 1;
		// 		// down.z = j * OG_SIZE;
		// 		// ogmsg.points.push_back(down);


		// 		back.x = i * OG_SIZE;
		// 		back.y = j * OG_SIZE;
		// 		back.z = -1;
		// 		ogmsg.points.push_back(back);
		// 	}
		
		ogmsg.header = cv_ptr->header;
		ogmsg.header.frame_id = FRAME_ID_CAMERA;
		ogmsg.header.stamp = ros::Time::now();

		world_ogmsg.header = cv_ptr->header;
		world_ogmsg.header.frame_id = FRAME_ID_WORLD;
		world_ogmsg.header.stamp = ros::Time::now();

		if (VISUALIZE_US_OG)
		{
			us_og_pub_.publish(ogmsg);
			us_world_og_pub_.publish(world_ogmsg);
		}

		ROS_INFO("<%u> time total %f", msg->header.seq, float(clock() - timer_total)/CLOCKS_PER_SEC);
	}

	void orbCb(const sensor_msgs::PointCloudConstPtr& msg)
	{
		// orbmsg = msg;
		orb_lock.lock();
		// if (!q_orbmsg.empty())
		// 	q_orbmsg.pop();
		q_orbmsg.push(msg);
		orb_lock.unlock();


	}

	void poseCb(const geometry_msgs::PoseStampedConstPtr& msg)
	{
		// posemsg = msg;
		pose_lock.lock();
		// if (!q_posemsg.empty())
		// 	q_posemsg.pop();
		q_posemsg.push(msg);
		pose_lock.unlock();
	}

	void scaleCb(const std_msgs::Float64ConstPtr& msg)
	{
		// if (msg->data <= 0){
		// 	world_og_lock.lock();
		// 	fix_world_og = !fix_world_og;
		// 	world_og_lock.unlock();
		// }else {
			mfscale = 1.0f / msg->data;
			reset_world_og();	
		// }
	}

	void fixOgCb(const std_msgs::Float64ConstPtr& msg)
	{
		world_og_lock.lock();
		if (msg->data > 0)
			fix_world_og = false;
		else
			fix_world_og = true;
		world_og_lock.unlock();
	}

	void draw_delaunay(cv::Mat& img, cv::Subdiv2D& subdiv)
	{
		vector<cv::Vec6f> triangleList;
		subdiv.getTriangleList(triangleList);
		vector<cv::Point> pt(3);
		cv::Size size = img.size();
		cv::Rect rect(0,0, size.width, size.height);

		for (size_t i = 0; i < triangleList.size(); i ++)
		{
			cv::Vec6f t = triangleList[i];
			pt[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
			pt[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
			pt[2] = cv::Point(cvRound(t[4]), cvRound(t[5]));

			// draw edges
			if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
			{
				// cv::Scalar color = cv::Scalar(0,0,0);
				cv::line(img, pt[0], pt[1], 0, 2);
				cv::line(img, pt[1], pt[2], 0, 2);
				cv::line(img, pt[2], pt[0], 0, 2);
			}
		}
	}

	void draw_points(cv::Mat& img, vector<cv::Point3f>& points, bool is_big = false)
	{
		for (vector<cv::Point3f>::iterator it = points.begin(); it != points.end(); it++)
		{
			cv::Point3f tmp = *it;
			cv::circle(img, cv::Point(cvRound(tmp.x), cvRound(tmp.y)), (is_big? 10 : 3), tmp.z, (is_big? 3 : -1));
		}
	}

	/*	locate a Point2f in the subdiv
		output a flag, as well as the vertex indices in the subdiv
	*/
	int locate_pt_in_subdiv(cv::Point2f pt, cv::Subdiv2D& subdiv, int& v1, int& v2, int& v3)
	{
		int edge = -1;
		int vertex = -1;
		int flag = subdiv.locate(pt, edge, vertex);

		switch (flag)
		{
			case cv::Subdiv2D::PTLOC_INSIDE:
			{
				v1 = subdiv.edgeOrg(edge);
				v2 = subdiv.edgeDst(edge);
				int next_edge_1 = subdiv.getEdge(edge, cv::Subdiv2D::PREV_AROUND_DST); //reversed eLnext
				// int next_edge_2 = subdiv.getEdge(edge, cv::Subdiv2D::NEXT_AROUND_ORG); //eOnext
				v3 = subdiv.edgeOrg(next_edge_1);

				return cv::Subdiv2D::PTLOC_INSIDE;
			}
			case cv::Subdiv2D::PTLOC_ON_EDGE: 
			{
				v1 = subdiv.edgeOrg(edge);
				v2 = subdiv.edgeDst(edge);

				return cv::Subdiv2D::PTLOC_ON_EDGE;
				// break;
			}
			case cv::Subdiv2D::PTLOC_VERTEX:
			{ 
				v1 = vertex;
				return cv::Subdiv2D::PTLOC_VERTEX;
				// break;
			}
			case cv::Subdiv2D::PTLOC_OUTSIDE_RECT:
			{
				return cv::Subdiv2D::PTLOC_OUTSIDE_RECT;
				// break;
			}
			default:
				ROS_INFO("ERROR in locate_pt_in_subdiv");
				break;
		}
		return cv::Subdiv2D::PTLOC_OUTSIDE_RECT;
	}

	int locate_pt_in_subdiv_opt(cv::Point2f pt, cv::Subdiv2D& subdiv, int& v1, int& v2, int& v3, int& v4, int& v5, int& v6)
	{
		int edge = -1;
		int vertex = -1;
		int flag = subdiv.locate(pt, edge, vertex);

		switch (flag)
		{
			case cv::Subdiv2D::PTLOC_INSIDE:
			{
				// std::pair<std::map<int,vector<int> >::iterator,bool> ret = dic.insert(pair<int, vector<int> > (edge, vector<int>()) );

				v1 = subdiv.edgeOrg(edge);
				v2 = subdiv.edgeDst(edge);
				int next_edge_1 = subdiv.getEdge(edge, cv::Subdiv2D::PREV_AROUND_DST); //reversed eLnext
				// int next_edge_2 = subdiv.getEdge(edge, cv::Subdiv2D::NEXT_AROUND_ORG); //eOnext
				v3 = subdiv.edgeOrg(next_edge_1);

				int next_edge_2 = subdiv.getEdge(edge, cv::Subdiv2D::PREV_AROUND_ORG);	// reversed eRnext
				v4 = subdiv.edgeDst(next_edge_2);

				int next_edge_3 = subdiv.getEdge(next_edge_1, cv::Subdiv2D::NEXT_AROUND_ORG); //eOnext of reversed eLnext
				v5 = subdiv.edgeDst(next_edge_3);

				int next_edge_4 = subdiv.getEdge(edge, cv::Subdiv2D::NEXT_AROUND_ORG); //eOnext
				int next_edge_5 = subdiv.getEdge(next_edge_4, cv::Subdiv2D::NEXT_AROUND_ORG); //eOnext of eOnext
				v6 = subdiv.edgeDst(next_edge_5);

				// output = ret.first->second;

				return cv::Subdiv2D::PTLOC_INSIDE;
			}
			case cv::Subdiv2D::PTLOC_ON_EDGE: 
			{
				// std::pair<std::map<int,vector<int> >::iterator,bool> ret = dic.insert(pair<int, vector<int> > (edge, vector<int>()) );

				v1 = subdiv.edgeOrg(edge);
				v2 = subdiv.edgeDst(edge);

				int next_edge_1 = subdiv.getEdge(edge, cv::Subdiv2D::PREV_AROUND_DST); //reversed eLnext
				v3 = subdiv.edgeOrg(next_edge_1);

				int next_edge_2 = subdiv.getEdge(edge, cv::Subdiv2D::PREV_AROUND_ORG);	// reversed eRnext
				v4 = subdiv.edgeDst(next_edge_2);

				// output = ret.first->second;

				return cv::Subdiv2D::PTLOC_ON_EDGE;
				// break;
			}
			case cv::Subdiv2D::PTLOC_VERTEX:
			{ 
				// std::pair<std::map<int,vector<int> >::iterator,bool> ret = dic.insert(pair<int, vector<int> > (edge, vector<int>()) );
				v1 = vertex;
				
				// set<int> set_neighbours;
				// pt_neighbours_in_subdiv(vertex, subdiv, set_neighbours);
				// v2 = set_neighbours.size();

				// for (set<int>::iterator it2 = set_neighbours.begin(); it2 != set_neighbours.end(); it2++)
				// {
				// 	ret.first->second.push_back(*it2);
				// }

				// output = ret.first->second;

				return cv::Subdiv2D::PTLOC_VERTEX;
				// break;
			}
			case cv::Subdiv2D::PTLOC_OUTSIDE_RECT:
			{
				return cv::Subdiv2D::PTLOC_OUTSIDE_RECT;
				// break;
			}
			default:
				ROS_INFO("ERROR in locate_pt_in_subdiv_opt");
				break;
		}
		return cv::Subdiv2D::PTLOC_OUTSIDE_RECT;
	}

	int locate_pt_in_subdiv2(cv::Point2f pt, cv::Subdiv2D& subdiv, vector<int>& output, map<int, vector<int> >& dic) //int& v1, int& v2, int& v3)
	{
		int edge = -1;
		int vertex = -1;
		int flag = subdiv.locate(pt, edge, vertex);

		map<int, vector<int> >::iterator it = dic.find(edge);
		if (it != dic.end())
		{
			output = it->second;
			return 666;
		}

		switch (flag)
		{
			case cv::Subdiv2D::PTLOC_INSIDE:
			{
				std::pair<std::map<int,vector<int> >::iterator,bool> ret = dic.insert(pair<int, vector<int> > (edge, vector<int>()) );

				ret.first->second.push_back(subdiv.edgeOrg(edge));
				ret.first->second.push_back(subdiv.edgeDst(edge));
				int next_edge_1 = subdiv.getEdge(edge, cv::Subdiv2D::PREV_AROUND_DST); //reversed eLnext
				// int next_edge_2 = subdiv.getEdge(edge, cv::Subdiv2D::NEXT_AROUND_ORG); //eOnext
				ret.first->second.push_back(subdiv.edgeOrg(next_edge_1));

				int next_edge_2 = subdiv.getEdge(edge, cv::Subdiv2D::PREV_AROUND_ORG);	// reversed eRnext
				ret.first->second.push_back(subdiv.edgeDst(next_edge_2));

				// int next_edge_3 = subdiv.getEdge(next_edge_1, cv::Subdiv2D::NEXT_AROUND_ORG); //eOnext of reversed eLnext
				// ret.first->second.push_back(subdiv.edgeDst(next_edge_3));

				// int next_edge_4 = subdiv.getEdge(edge, cv::Subdiv2D::NEXT_AROUND_ORG); //eOnext
				// int next_edge_5 = subdiv.getEdge(next_edge_4, cv::Subdiv2D::NEXT_AROUND_ORG); //eOnext of eOnext
				// ret.first->second.push_back(subdiv.edgeDst(next_edge_5));

				output = ret.first->second;

				return cv::Subdiv2D::PTLOC_INSIDE;
			}
			case cv::Subdiv2D::PTLOC_ON_EDGE: 
			{
				std::pair<std::map<int,vector<int> >::iterator,bool> ret = dic.insert(pair<int, vector<int> > (edge, vector<int>()) );

				ret.first->second.push_back(subdiv.edgeOrg(edge));
				ret.first->second.push_back(subdiv.edgeDst(edge));

				int next_edge_1 = subdiv.getEdge(edge, cv::Subdiv2D::PREV_AROUND_DST); //reversed eLnext
				ret.first->second.push_back(subdiv.edgeOrg(next_edge_1));

				int next_edge_2 = subdiv.getEdge(edge, cv::Subdiv2D::PREV_AROUND_ORG);	// reversed eRnext
				ret.first->second.push_back(subdiv.edgeDst(next_edge_2));

				output = ret.first->second;

				return cv::Subdiv2D::PTLOC_ON_EDGE;
				// break;
			}
			case cv::Subdiv2D::PTLOC_VERTEX:
			{ 
				std::pair<std::map<int,vector<int> >::iterator,bool> ret = dic.insert(pair<int, vector<int> > (edge, vector<int>()) );

				ret.first->second.push_back(vertex);
				set<int> set_neighbours;
				pt_neighbours_in_subdiv(vertex, subdiv, set_neighbours);
				for (set<int>::iterator it2 = set_neighbours.begin(); it2 != set_neighbours.end(); it2++)
				{
					ret.first->second.push_back(*it2);
				}

				output = ret.first->second;

				return cv::Subdiv2D::PTLOC_VERTEX;
				// break;
			}
			case cv::Subdiv2D::PTLOC_OUTSIDE_RECT:
			{
				return cv::Subdiv2D::PTLOC_OUTSIDE_RECT;
				// break;
			}
			default:
				ROS_INFO("ERROR in locate_pt_in_subdiv3");
				break;
		}
		return cv::Subdiv2D::PTLOC_OUTSIDE_RECT;
	}

	void pt_neighbours_in_subdiv(int pt, cv::Subdiv2D& subdiv, set<int>& set_neighbours, bool are_default_pts_excluded = false)
	{
		// ROS_INFO("%d starts adding neighbours", pt);
		int edge;
		subdiv.getVertex(pt, &edge);
		if(pt != subdiv.edgeOrg(edge))
			edge = subdiv.rotateEdge(edge, 2);

		int neighbour_pt = subdiv.edgeDst(edge);
		while(set_neighbours.find(neighbour_pt) == set_neighbours.end())
		{
			set_neighbours.insert(neighbour_pt);
			// ROS_INFO("%d has neighbour %d", pt, neighbour_pt);
			edge = subdiv.nextEdge(edge);
			neighbour_pt = subdiv.edgeDst(edge);
		}
		// ROS_INFO("# of set_neighbours is %ld", set_neighbours.size());
		if (are_default_pts_excluded)
			for (int i = 0; i < 4; i ++)
				set_neighbours.erase(i);
	}

	bool is_outlier(float value, vector<float>& input)
	{
		int size_of_vec = input.size();
		sort(input.begin(), input.end());
		float q1 = input[size_of_vec / 4];
		float q2 = input[size_of_vec / 2];
		float q3 = input[3 * size_of_vec / 4];
		float iqr = q3 - q1;

		float lower_fence = q1 - 1.5 * iqr;
		float upper_fence = q3 + 1.5 * iqr;

		return (value < lower_fence || value > upper_fence);
	}

	float look_up_pt_inv_log_mega_depth(cv::Point2f pt, cv::Mat mat_mega)
	{
		return mat_mega.at<float>(cvRound(pt.y), cvRound(pt.x));
	}


	float squared_diff_in_log_mega_depth_from(float log_depth, cv::Point2f pt, cv::Mat mat_mega)
	{
		if(fabs(pt.x) >= MEGA_WIDTH || fabs(pt.y) >= MEGA_WIDTH)
			return (log_depth - 1.f) * (log_depth - 1.f);	// assume the default mega_depth is 1, i.e., the nearest possible

		float log_mega_depth = 1.f / look_up_pt_inv_log_mega_depth(pt, mat_mega);
		return (log_depth - log_mega_depth) * (log_depth - log_mega_depth);
	}

	float squared_dist_between_pts(cv::Point2f pt1, cv::Point2f pt2)
	{
		return (pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y);
	}

	float max_in_vector(vector<float>& v)
	{
		float result = v[0];
		for (vector<float>::iterator it = v.begin(); it != v.end(); it ++)
		{
			if (result < *it)
				result = *it;
		}
		return result;
	}

	float sum_of_vector(vector<float>& v)
	{
		float result = 0.f;
		for (vector<float>::iterator it = v.begin(); it != v.end(); it ++)
		{
			result += *it;
		}
		return result;
	}

	float compute_weighted_log_scale(vector<float> vec_log_scale, vector<float> vec_squared_diff)
	{
		int size = vec_log_scale.size();

		float weighted_scale_sum = 0.f;
		float weight_sum = 0.f;
		for(int idx = 0; idx < size; idx++)
		{
			float weight = exp(- vec_squared_diff[idx] / SQUARED_SIGMA_R);
			weighted_scale_sum += (exp(vec_log_scale[idx]) * weight);
			weight_sum += weight;
		}
		return log(weighted_scale_sum / weight_sum);

		// int size = vec_log_scale.size();

		// bool has_in = false;
		// vector<int> vec_in_idx;

		// for (int idx = 0; idx < size; idx++)
		// {
		// 	if (vec_squared_diff[idx] < 3 * SIGMA)
		// 	{
		// 		has_in = true;
		// 		vec_in_idx.push_back(idx);
		// 	}
		// }

		// if (has_in)
		// {
		// 	float weighted_scale_sum = 0.f;
		// 	float weight_sum = 0.f;
		// 	for(vector<int>::iterator it = vec_in_idx.begin(); it != vec_in_idx.end(); it++)
		// 	{
		// 		float weight = exp(- vec_squared_diff[*it] / SQUARED_SIGMA_R);
		// 		weighted_scale_sum += (exp(vec_log_scale[*it]) * weight);
		// 		weight_sum += weight;
		// 	}

		// 	return log(weighted_scale_sum / weight_sum);
		// } 
		// // all are outside
		// else
		// {
		// 	float scale_sum = 0.f;
		// 	for(int idx = 0; idx < size; idx++)
		// 	{
		// 		scale_sum += exp(vec_log_scale[idx]);
		// 	}

		// 	return scale_sum / size;
		// }
	}

	void union_of_sets(set<int>& input, set<int>& output)
	{
		for(set<int>::iterator it = input.begin(); it != input.end(); it++)
		{
			output.insert(*it);
		}
	}

	void apply_Rt_to_pt(float x, float y, float z, float tx, float ty, float tz, float a, float b, float c, float d, float& x_out, float& y_out, float& z_out)
	{
		cv::Mat mat_R(3, 3, CV_32FC1);
		mat_R.at<float>(0, 0) = a*a + b*b - c*c - d*d;
		mat_R.at<float>(0, 1) = 2*b*c - 2*a*d;
		mat_R.at<float>(0, 2) = 2*b*d + 2*a*c;
		
		mat_R.at<float>(1, 0) = 2*b*c + 2*a*d;
		mat_R.at<float>(1, 1) = a*a - b*b + c*c - d*d;
		mat_R.at<float>(1, 2) = 2*c*d - 2*a*b;
		
		mat_R.at<float>(2, 0) = 2*b*d - 2*a*c;
		mat_R.at<float>(2, 1) = 2*c*d + 2*a*b;
		mat_R.at<float>(2, 2) = a*a - b*b - c*c + d*d;

		cv::Mat mat_T(3, 1, CV_32FC1);
		mat_T.at<float>(0,0) = tx;
		mat_T.at<float>(1,0) = ty;
		mat_T.at<float>(2,0) = tz;

		cv::Mat mat_pt(3, 1, CV_32FC1);
		mat_pt.at<float>(0,0) = x;
		mat_pt.at<float>(1,0) = y;
		mat_pt.at<float>(2,0) = z;

		cv::Mat mat_out = mat_R * mat_pt + mat_T;
		x_out = mat_out.at<float>(0,0);
		y_out = mat_out.at<float>(1,0);
		z_out = mat_out.at<float>(2,0);
	}

	void posemsg_to_mat(geometry_msgs::Pose posemsg, cv::Mat& mat)
	{
		// rotation
		float a = posemsg.orientation.w;
		float b = posemsg.orientation.x;
		float c = posemsg.orientation.y;
		float d = posemsg.orientation.z;

		mat.at<float>(0,0) = a*a + b*b - c*c - d*d;
		mat.at<float>(0,1) = 2*b*c - 2*a*d;
		mat.at<float>(0,2) = 2*b*d + 2*a*c;
		
		mat.at<float>(1,0) = 2*b*c + 2*a*d;
		mat.at<float>(1,1) = a*a - b*b + c*c - d*d;
		mat.at<float>(1,2) = 2*c*d - 2*a*b;
		
		mat.at<float>(2,0) = 2*b*d - 2*a*c;
		mat.at<float>(2,1) = 2*c*d + 2*a*b;
		mat.at<float>(2,2) = a*a - b*b - c*c + d*d;

		// translation
		mat.at<float>(0,3) = posemsg.position.x * mfscale;
		mat.at<float>(1,3) = posemsg.position.y * mfscale;
		mat.at<float>(2,3) = posemsg.position.z * mfscale;

		mat.at<float>(3,0) = 0;
		mat.at<float>(3,1) = 0;
		mat.at<float>(3,2) = 0;
		mat.at<float>(3,3) = 1;
	}

	void positionmsg_to_mat(geometry_msgs::Point positionmsg, cv::Mat& mat)
	{
		// translation
		mat.at<float>(0,0) = positionmsg.x;
		mat.at<float>(1,0) = positionmsg.y;
		mat.at<float>(2,0) = positionmsg.z;
		mat.at<float>(3,0) = 1;
	}

	bool computeCollisionFreePose(upsample::CollisionAvoidance::Request &req,
								  upsample::CollisionAvoidance::Response &res)
	{
		// // OG coordinate (reference coordinate)
		// cv::Mat mat_reference(4, 4, CV_32FC1);
		// posemsg_to_mat(posemsg->pose, mat_reference);

		// points in the world cooridnate
		cv::Mat vec_target(4, 1, CV_32FC1);
		positionmsg_to_mat(req.t_position, vec_target);

		cv::Mat vec_current(4, 1, CV_32FC1);
		positionmsg_to_mat(req.c_position, vec_current);

		// // points in the reference coordinate
		// vec_target = mat_reference.inv() * vec_target;
		// vec_current = mat_reference.inv() * vec_current;

		float xx = vec_target.at<float>(0,0) - vec_current.at<float>(0,0);
		float yy = vec_target.at<float>(1,0) - vec_current.at<float>(1,0);
		float zz = vec_target.at<float>(2,0) - vec_current.at<float>(2,0);

		// the direction of proceed
		cv::Mat vec_proceed(3, 1, CV_32FC1);
		vec_proceed.at<float>(0,0) = xx * ATTRACTIVE_FORCE;
		vec_proceed.at<float>(1,0) = yy * ATTRACTIVE_FORCE;
		vec_proceed.at<float>(2,0) = zz * ATTRACTIVE_FORCE;

		cv::Mat vec_repusive(3, 1, CV_32FC1);
		vec_repusive.at<float>(0,0) = 0;
		vec_repusive.at<float>(1,0) = 0;
		vec_repusive.at<float>(2,0) = 0;

		world_og_lock.lock();
		for (map<int,float>::iterator it = world_og_score.begin(); it != world_og_score.end(); it++)
		{
			if (it->second < 0.01)
				continue;
			
			int xxx = it->first / (NUM_WORLD_Y_GRIDS * NUM_WORLD_Z_GRIDS);
			int yyy = (it->first - xxx * NUM_WORLD_Y_GRIDS * NUM_WORLD_Z_GRIDS) / NUM_WORLD_Z_GRIDS;
			int zzz = it->first % NUM_WORLD_Z_GRIDS;

			float dx = vec_current.at<float>(0,0) - (xxx * OG_SIZE - OG_CLIP_WORLD + 0.5 * OG_SIZE);
			float dy = vec_current.at<float>(1,0) - (yyy * OG_SIZE - OG_CLIP_WORLD + 0.5 * OG_SIZE);
			float dz = vec_current.at<float>(2,0) - (zzz * OG_SIZE - OG_CLIP_WORLD + 0.5 * OG_SIZE);

			float dist = sqrt(dx * dx + dy * dy + dz * dz);

			if (dist > 1)
				continue;

			float force = REPULSIVE_FORCE / dist * it->second;	// note that it->second is the score

			vec_repusive.at<float>(0,0) += force * dx;
			vec_repusive.at<float>(1,0) += force * dy;
			vec_repusive.at<float>(2,0) += force * dz;
		}
		world_og_lock.unlock();

		// for (int i = 0; i < 3; i++)
		// {
		// 	if (vec_repusive.at<float>(i,0) > 1.f)
		// 		vec_repusive.at<float>(i,0) = 1; 
		// 	if (vec_repusive.at<float>(i,0) < -1.f)
		// 		vec_repusive.at<float>(i,0) = -1;
		// }


		// the resulting direction
		cv::Mat vec_move = vec_proceed + vec_repusive;

		// a negative dot product implies the resulting direction is reversed
		// cv::Mat dot_product = vec_proceed.t() * vec_move;
		// if (dot_product.at<float>(0,0) < 0)
		// {
		// 	vec_move.at<float>(0,0) = 0;
		// 	vec_move.at<float>(1,0) = 0;
		// 	vec_move.at<float>(2,0) = 0;
		// }

		// for (int i = 0; i < 3; i++)
		// {
		// 	if (vec_move.at<float>(i,0) > 1.f)
		// 		vec_move.at<float>(i,0) = 1;
		// 	if (vec_move.at<float>(i,0) < -1.f)
		// 		vec_move.at<float>(i,0) = -1;
		// }

		cv::Mat vec_result(3, 1, CV_32FC1);
		vec_result.at<float>(0,0) = vec_current.at<float>(0,0) + vec_move.at<float>(0,0) * PROGRESSIVE_RATIO;
		vec_result.at<float>(1,0) = vec_current.at<float>(1,0) + vec_move.at<float>(1,0) * PROGRESSIVE_RATIO;
		vec_result.at<float>(2,0) = vec_current.at<float>(2,0) + vec_move.at<float>(2,0) * PROGRESSIVE_RATIO;
		// vec_result.at<float>(3,0) = 1;


		if (VISUALIZE_US_GOAL)
		{
			geometry_msgs::PointStamped current_position_msg, goal_position_msg, intermediate_position_msg;
			current_position_msg.header = posemsg->header;
			current_position_msg.header.frame_id = FRAME_ID_WORLD;

			current_position_msg.point.x = vec_current.at<float>(0,0);
			current_position_msg.point.y = vec_current.at<float>(1,0);
			current_position_msg.point.z = vec_current.at<float>(2,0);
			current_position_pub_.publish(current_position_msg);

			intermediate_position_msg.header = posemsg->header;
			intermediate_position_msg.header.frame_id = FRAME_ID_WORLD;

			intermediate_position_msg.point.x = vec_result.at<float>(0,0);
			intermediate_position_msg.point.y = vec_result.at<float>(1,0);
			intermediate_position_msg.point.z = vec_result.at<float>(2,0);
			intermediate_position_pub_.publish(intermediate_position_msg);


			goal_position_msg.header = posemsg->header;
			goal_position_msg.header.frame_id = FRAME_ID_WORLD;

			goal_position_msg.point.x = vec_target.at<float>(0,0);
			goal_position_msg.point.y = vec_target.at<float>(1,0);
			goal_position_msg.point.z = vec_target.at<float>(2,0);
			goal_position_pub_.publish(goal_position_msg);
		}

		// vec_result = mat_reference * vec_result;

		res.i_position.x = vec_result.at<float>(0,0);
		res.i_position.y = vec_result.at<float>(1,0);
		res.i_position.z = vec_result.at<float>(2,0);

		return true;
	}

	float len_of_vec(cv::Mat mat)
	{
		cv::Mat squared_len = mat.t() * mat;
		return sqrt(squared_len.at<float>(0,0));
	}
};


int main(int argc, char **argv)
{
	ros::init(argc, argv, "upsample_node");

	UpSampler us;

	ros::NodeHandle nh_;
	us.mega_sub_ = nh_.subscribe("/mega/inv_depth", 10, &UpSampler::imageCb, &us);
	us.orb_sub_ = nh_.subscribe("/orb_slam/pcl_cam", 30, &UpSampler::orbCb, &us);
	us.pose_sub_ = nh_.subscribe("/orb_slam/unscaled_pose", 30, &UpSampler::poseCb, &us);
	us.scale_sub_ = nh_.subscribe("/ml_scale", 1, &UpSampler::scaleCb, &us);
	us.fix_og_sub_ = nh_.subscribe("/us/fix_og", 1, &UpSampler::fixOgCb, &us);

	us.mega_pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud>("/us/mega_raw", 1);
	us.orb_pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud>("/us/orb_raw", 1);
	us.us_pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud>("/us/pcl", 1);
	us.us_og_pub_ = nh_.advertise<sensor_msgs::PointCloud>("/us/og", 1);
	us.tri_img_pub_ = nh_.advertise<sensor_msgs::Image>("/us/triangulate_raw", 1);
	us.scale_img_pub_ = nh_.advertise<sensor_msgs::Image>("/us/scale_raw", 1);
	us.us_img_pub_ = nh_.advertise<sensor_msgs::Image>("/us/img", 1);

	us.us_world_og_pub_ = nh_.advertise<sensor_msgs::PointCloud>("/us/og_world", 1);

	// another node handle and its callback queue and spinner
	ros::NodeHandle srv_nh_;
	ros::CallbackQueue srv_queue_;
	srv_nh_.setCallbackQueue(&srv_queue_);
	us.service_ = srv_nh_.advertiseService("/us/collision_avoidance", &UpSampler::computeCollisionFreePose, &us);

	us.current_position_pub_ = srv_nh_.advertise<geometry_msgs::PointStamped>("/us/current_position", 1);
	us.goal_position_pub_ = srv_nh_.advertise<geometry_msgs::PointStamped>("/us/goal_position", 1);
	us.intermediate_position_pub_ = srv_nh_.advertise<geometry_msgs::PointStamped>("/us/intermediate_position", 1);

	ros::AsyncSpinner async_spinner(1, &srv_queue_);	// spawn async spinner with 1 thread, running on our custom queue
	async_spinner.start();								// start the spinner

	// the default node handle
	ros::spin();


	return 0;
}
