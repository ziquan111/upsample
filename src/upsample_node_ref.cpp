#include <ros/ros.h>
#include <std_msgs/String.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <geometry_msgs/Point32.h>
#include <sensor_msgs/PointCloud.h>

#include <sstream>
// #include <mutex>
#include <time.h>

#define Ku 338.7884
#define Kv 444.1741
#define U0 256.0
#define V0 192.0

#define ORB_WIDTH 856.f
#define ORB_HEIGHT 480.f
#define MEGA_WIDTH 512.f
#define MEGA_HEIGHT 384.f

#define SQUARED_SIGMA_S 100.f
#define SIGMA 50.f
#define SQUARED_SIGMA_R (SIGMA * SIGMA)

#define FRAME_ID_WORLD "odom"
#define FRAME_ID_CAMERA "base_link"

#define PATCH_HALF_SIZE 5

#define DO_BLUR false

// std::mutex pcl_mutex;

using namespace std;

class UpSampler
{
public:
	// ros::NodeHandle nh_;
	// image_transport::ImageTransport it_;
	ros::Subscriber mega_sub_, orb_sub_;
	ros::Publisher mega_pcl_pub_, orb_pcl_pub_, us_pcl_pub_, tri_img_pub_, scale_img_pub_, us_img_pub_;
	sensor_msgs::PointCloudPtr orbmsg;
	// std::lock_guard<std::mutex> lock(pcl_mutex)

	UpSampler()
	{
	}

	~UpSampler()
	{
	}

	void imageCb(const sensor_msgs::ImageConstPtr& msg)
	{
		// ROS_INFO("got a message");
		cv_bridge::CvImagePtr cv_ptr, cv_ptr_copy_tri, cv_ptr_copy_scale, cv_ptr_copy_us;
		try
		{
			cv_ptr = cv_bridge::toCvCopy(msg, "32FC1");
			cv_ptr_copy_tri = cv_bridge::toCvCopy(msg, "32FC1");
			cv_ptr_copy_scale = cv_bridge::toCvCopy(msg, "32FC1");
			cv_ptr_copy_us = cv_bridge::toCvCopy(msg, "32FC1");
			// cv::cvtColor(cv_ptr_bgr->image, cv_ptr_bgr->image, CV_GRAY2BGR);
		}
		catch(cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}

		// ROS_INFO("cv_ptr init");
		sensor_msgs::PointCloud pclMsg;
		// float min_z = 99999.f;

		for (size_t v = 0; v < cv_ptr->image.rows; v++)
			for (size_t u = 0; u < cv_ptr->image.cols; u++)
			{
				// ROS_INFO("u at %d, v at %d", u, v);
				geometry_msgs::Point32 pt;
				pt.z = exp(1.f/cv_ptr->image.at<float>(v,u));	// x and y depend on z
				pt.x = (u - U0) / Ku * pt.z;
				pt.y = (v - V0) / Kv * pt.z;
				// pt.z = 1.f/cv_ptr->image.at<float>(v,u);
				pclMsg.points.push_back(pt);
				// if (min_z > pt.z)
				// 	min_z = pt.z;
			}
		pclMsg.header = cv_ptr->header;
		pclMsg.header.frame_id = FRAME_ID_CAMERA;

		mega_pcl_pub_.publish(pclMsg);
		// ROS_INFO("min_z is %f which is always 1", min_z);

		// 
		// repub orb
		// 

		if(orbmsg == NULL)
			return;
		int num_points = orbmsg->points.size();
		ROS_INFO("<%u> processing orb message---, %d", msg->header.seq, num_points);
		orb_pcl_pub_.publish(orbmsg);
		// ROS_INFO("orbmsg: %f, %f", orbmsg->channels[0].values[0], orbmsg->channels[1].values[0]);

		// 
		// upsampling
		// 

		// full image approach : too slow...
		// cv::Mat mat_scale(MEGA_HEIGHT, MEGA_WIDTH, CV_32FC1, -1); // default scale is set to be -1
		// float scale_max = -1.f;
		// // cv::Mat mat_orb_depth(MEGA_HEIGHT, MEGA_WIDTH, CV_32FC1, -1); // default depth is set to be -1
		// vector<cv::Point3f> vec_orb_depth;	// image coordinate and depth
		// for (int i = 0; i < num_points; i++)
		// {
		// 	// use the u, v channels
		// 	float tmp_u = (orbmsg->channels[0].values[i] / ORB_WIDTH * MEGA_WIDTH);
		// 	float tmp_v = (orbmsg->channels[1].values[i] / ORB_HEIGHT * MEGA_HEIGHT);
		// 	// mat_orb_depth.at<float>(cvRound(tmp_v), cvRound(tmp_u)) = orbmsg->points[i].z;	// fill in the depth
		// 	vec_orb_depth.push_back(cv::Point3f(tmp_v, tmp_u, orbmsg->points[i].z));	// fill in the vector of points
		// }

		// for (size_t v = 0; v < cv_ptr->image.rows; v++)
		// {
		// 	for (size_t u = 0; u < cv_ptr->image.cols; u++)
		// 	{
		// 		float weighted_scale = 0.f;
		// 		float weight_sum = 0.f;
		// 		for(vector<cv::Point3f>::iterator it = vec_orb_depth.begin(); it != vec_orb_depth.end(); it++)
		// 		{
		// 			cv::Point3f pt_depth = *it;
		// 			cv::Point2f tmp_pt(pt_depth.x, pt_depth.y);
		// 			cv::Point2f pt(v, u);
		// 			float squared_dis = squared_dist_between_pts(pt, tmp_pt);

		// 			if (squared_dis > SQUARED_SIGMA_S * 4)
		// 				continue;

		// 			float pt_inv_depth = look_up_pt_inv_mega_depth(tmp_pt, cv_ptr->image);
		// 			float pt_orb_depth = pt_depth.z;
		// 			float scale = pt_inv_depth * pt_orb_depth;
		// 			float weight = exp(- squared_dis / SQUARED_SIGMA_S);
					
		// 			weighted_scale += (scale * weight);
		// 			weight_sum += weight;
		// 		}
		// 		weighted_scale /= weight_sum;
		// 		mat_scale.at<float>(v, u) = weighted_scale;
		// 		if(scale_max < weighted_scale)
		// 			scale_max = weighted_scale;
		// 	}
		// }

		// for (size_t v = 0; v < cv_ptr_copy_scale->image.rows; v++)
		// 	for (size_t u = 0; u < cv_ptr_copy_scale->image.cols; u++)
		// 	{
		// 		cv_ptr_copy_scale->image.at<float>(v,u) = mat_scale.at<float>(v,u) / scale_max;
		// 	}
		// scale_img_pub_.publish(cv_ptr_copy_scale->toImageMsg());

		// triangluate approach


		// step 1: perform the delaunary, estimate a (weighted) average scale, and prepare smoothing patches
		// prepare for delaunary
		cv::Rect rect(0,0, int(MEGA_WIDTH), int(MEGA_HEIGHT));
		cv::Subdiv2D subdiv(rect);

		// prepare for orb_depth look up
		cv::Mat mat_orb_depth(MEGA_HEIGHT, MEGA_WIDTH, CV_32FC1, -1); // default depth is set to be -1
		
		// prepare for average scale		
		float scale_average = 0.f;
		float weight_sum = 0.f;

		// prepare for smoothing patches

		int num_unique_points = num_points + 4;	// including the virual points used by Subdiv2D
		int max_pt_id = 0;
		// vector<vector<float> > vec_scale_patches(num_unique_points);
		vector<float> 		vec_scale_sparse(num_unique_points);
		// vector<float> vec_scale_patches_average(num_points + 4);
		
		// // vector<cv::Point2f> points;
		for (int i = 0; i < num_points; i++)
		{
			// use the u, v channels
			float tmp_u = (orbmsg->channels[0].values[i] / ORB_WIDTH * MEGA_WIDTH);
			float tmp_v = (orbmsg->channels[1].values[i] / ORB_HEIGHT * MEGA_HEIGHT);
			float tmp_orb_depth = orbmsg->points[i].z;
			cv::Point2f tmp_pt(tmp_u,tmp_v);

			// use the pt_id to refer a patch
			int pt_id = subdiv.insert(tmp_pt);

			// existing pt_id
			if (pt_id <= max_pt_id)
			{
				num_unique_points --;
				// vec_scale_patches.resize(num_unique_points);
				vec_scale_sparse.resize(num_unique_points);
				// existing depth is closer than the new one, do nothing
				if (mat_orb_depth.at<float>(cvRound(tmp_v), cvRound(tmp_u)) <= tmp_orb_depth)
				{
					continue;
				} 
				// existing depth is further then the new one, replace the old one
				// else {
				// 	vec_scale_patches[pt_id].clear();
				// }
			}
			else{
				max_pt_id = pt_id;
			}

			float tmp_scale = tmp_orb_depth * look_up_pt_inv_mega_depth(tmp_pt, cv_ptr->image);
			vec_scale_sparse[pt_id] = tmp_scale;
			// ROS_INFO("<%u> pt_id %d : scale %f", msg->header.seq, pt_id, tmp_scale);

			scale_average +=  tmp_scale / tmp_orb_depth;	// scale * weight
			weight_sum += 1.f / tmp_orb_depth;
			// ROS_INFO("pt %d with pt_id  %d at (%f, %f)", i, pt_id, tmp_u, tmp_v);
		}
		scale_average /= weight_sum;
		for (int pt_id = 0; pt_id < 4; pt_id++)
		{
				vec_scale_sparse[pt_id] = scale_average;
				continue;
		}

		// step 2.2: compute the scale
		
		float time_used_to_locate = 0.f;
		float time_used_to_prepare = 0.f;
		float time_used_to_compute = 0.f;
		clock_t timer;

		float scale_max = -1.f, scale_min = 9999.f;
		float scale_sum = 0.f;
		float scale_count = 0.f;
		cv::Mat mat_scale(MEGA_HEIGHT, MEGA_WIDTH, CV_32FC1, -1); // default scale is set to be -1
// #pragma omp parallel
// #pragma omp for
		for (size_t v = 0; v < cv_ptr->image.rows; v++)
		{
			for (size_t u = 0; u < cv_ptr->image.cols; u++)
			{
				timer = clock();
				int v1,v2,v3;
				int flag = locate_pt_in_subdiv(cv::Point2f(u, v), subdiv, v1, v2, v3);
				timer = clock() - timer;
				time_used_to_locate += ((float)timer)/CLOCKS_PER_SEC;

				if(flag == cv::Subdiv2D::PTLOC_INSIDE)
				{
					timer = clock();

					cv::Point2f pt1 = subdiv.getVertex(v1);
					cv::Point2f pt2 = subdiv.getVertex(v2);
					cv::Point2f pt3 = subdiv.getVertex(v3);


					// compute the scale: orb_depth / mega_depth
					float scale1 = vec_scale_sparse[v1];
					float scale2 = vec_scale_sparse[v2];
					float scale3 = vec_scale_sparse[v3];

					// compute the squared distances
					cv::Point2f pt(u, v);
					// float squared_dis1 = squared_dist_between_pts(pt, pt1);
					// float squared_dis2 = squared_dist_between_pts(pt, pt2);
					// float squared_dis3 = squared_dist_between_pts(pt, pt3);

					// compute the squared differences in mega_depth
					float depth = 1.f/look_up_pt_inv_mega_depth(pt, cv_ptr->image);
					float squared_diff1 = squared_diff_in_mega_depth_from(depth, pt1, cv_ptr->image); //(depth - 1.f/v1_inv_mega_depth) * (depth - 1.f/v1_inv_mega_depth);
					float squared_diff2 = squared_diff_in_mega_depth_from(depth, pt2, cv_ptr->image); //(depth - 1.f/v2_inv_mega_depth) * (depth - 1.f/v2_inv_mega_depth);
					float squared_diff3 = squared_diff_in_mega_depth_from(depth, pt3, cv_ptr->image); //(depth - 1.f/v3_inv_mega_depth) * (depth - 1.f/v3_inv_mega_depth);
					// ROS_INFO("squared_diff are %.6f, %.6f, %.6f", squared_diff1, squared_diff2, squared_diff3);

					timer = clock() - timer;
					time_used_to_prepare += ((float)timer)/CLOCKS_PER_SEC;
					// compute weights
					// float weight1 = exp(- squared_diff1 / SQUARED_SIGMA_R);//exp(- squared_dis1 / SQUARED_SIGMA_S);// * exp(- squared_dep1 / SQUARED_SIGMA_R);
					// float weight2 = exp(- squared_diff2 / SQUARED_SIGMA_R);//exp(- squared_dis2 / SQUARED_SIGMA_S);// * exp(- squared_dep2 / SQUARED_SIGMA_R);
					// float weight3 = exp(- squared_diff3 / SQUARED_SIGMA_R);//exp(- squared_dis3 / SQUARED_SIGMA_S);// * exp(- squared_dep3 / SQUARED_SIGMA_R);
					// float weight_normalizer += (weight1 + weight2 + weight3);

					timer = clock();
					// compute scale
					//float scale = exp( log(weight1 * scale1 + weight2 * scale2 + weight3 * scale3) - log(weight1 + weight2 + weight3));
					float scale = compute_weighted_scale(scale1, scale2, scale3, squared_diff1, squared_diff2, squared_diff3); //(weight1 * scale1 + weight2 * scale2 + weight3 * scale3) / (weight1 + weight2 + weight3);
					// ROS_INFO("the weight is %.8f", (weight1 + weight2 + weight3));

					if (scale1 < 0)
						ROS_ERROR("<%u> scale1 is negative %f",msg->header.seq, scale1);
					if (scale2 < 0)
						ROS_ERROR("<%u> scale2 is negative %f",msg->header.seq, scale2);
					if (scale3 < 0)
						ROS_ERROR("<%u> scale3 is negative %f",msg->header.seq, scale3);
					mat_scale.at<float>(v, u) = scale;
					scale_sum += scale;
					scale_count += 1.f;
					if (scale_max < scale)
						scale_max = scale;
					if (scale_min > scale)
					{
						scale_min = scale;
						// ROS_INFO("<%u> new scale_min %f, because of pt_id %d, %d, %d",msg->header.seq, scale, v1, v2, v3);
					}
					timer = clock() - timer;
					time_used_to_compute += ((float)timer)/CLOCKS_PER_SEC;
				}
			}
		}
		// timer = clock() - timer;
		// ROS_INFO("TIME USED: %f", ((float)timer)/CLOCKS_PER_SEC);z
		// scale_max /= weight_normalizer;
		// scale_sum /= weight_normalizer;
		// for (size_t v = 0; v < MEGA_HEIGHT; v++)
		// 	for (size_t u = 0; u < MEGA_WIDTH; u++)
		// 		if(mat_scale.at<float>(v,u) < 0)
		// 			mat_scale.at<float>(v,u) = 0;//scale_sum / scale_count;
				// else
					// mat_scale.at<float>(v,u) /= weight_normalizer;
		if (DO_BLUR)
			cv::GaussianBlur(mat_scale, mat_scale, cv::Size(0,0), 10);
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(mat_scale, &minVal, &maxVal, &minLoc, &maxLoc);
		ROS_INFO("scale_sum is %f, scale_count is %f, scale_max is %f, (%lf), scale_min is %f, (%lf), average is %f", scale_sum, scale_count, scale_max, maxVal, scale_min, minVal, scale_sum / scale_count);
		ROS_INFO("time to locate %f, time to prepare %f, time to compute %f", time_used_to_locate, time_used_to_prepare, time_used_to_compute);

		// visualize triangulation
		vector<cv::Point3f> points_to_draw;
		// float max_scale_patch_average = max_in_vector(vec_scale_patches_average);
		float max_scale_sparse = max_in_vector(vec_scale_sparse);
		for (int i = 0; i < num_unique_points; i ++)
		{
			cv::Point2f tmp_pt = subdiv.getVertex(i) ;
			points_to_draw.push_back(cv::Point3f(tmp_pt.x, tmp_pt.y, vec_scale_sparse[i] / max_scale_sparse));
		}
		// for (int i = 0; i < num_points; i++)
		// {
		// 	// use the u, v channels
		// 	float tmp_u = (orbmsg->channels[0].values[i] / ORB_WIDTH * MEGA_WIDTH);
		// 	float tmp_v = (orbmsg->channels[1].values[i] / ORB_HEIGHT * MEGA_HEIGHT);
		// 	points_to_draw.push_back(cv::Point3f(tmp_u, tmp_v, orbmsg->points[i].z / scale_max));
		// }
		draw_delaunay(cv_ptr_copy_tri->image, subdiv);
		draw_points(cv_ptr_copy_tri->image, points_to_draw);
		tri_img_pub_.publish(cv_ptr_copy_tri->toImageMsg());

		// visualize the scales
		for (size_t v = 0; v < cv_ptr_copy_scale->image.rows; v++)
			for (size_t u = 0; u < cv_ptr_copy_scale->image.cols; u++)
			{
				if(mat_scale.at<float>(v,u) < 0)
					cv_ptr_copy_scale->image.at<float>(v,u) = 0;//mat_scale.at<float>(v,u) / scale_max;
				else
					cv_ptr_copy_scale->image.at<float>(v,u) = mat_scale.at<float>(v,u) / maxVal;
			}
		scale_img_pub_.publish(cv_ptr_copy_scale->toImageMsg());

		// visualize the upsampled output
		float max_inv_scale = -1.f;
		for (size_t v = 0; v < cv_ptr_copy_us->image.rows; v++)
			for (size_t u = 0; u < cv_ptr_copy_us->image.cols; u++)
			{
				if(mat_scale.at<float>(v,u) < 0)
					cv_ptr_copy_us->image.at<float>(v,u) = 0;//mat_scale.at<float>(v,u) / scale_max;
				else
					cv_ptr_copy_us->image.at<float>(v,u) = log(mat_scale.at<float>(v,u)) + (1.f/ cv_ptr_copy_us->image.at<float>(v,u));
					// cv_ptr_copy_us->image.at<float>(v,u) = 1.f / (1.f / cv_ptr_copy_us->image.at<float>(v,u) + log(mat_scale.at<float>(v,u)) / log(10.f));
				if(max_inv_scale < cv_ptr_copy_us->image.at<float>(v,u))
					max_inv_scale = cv_ptr_copy_us->image.at<float>(v,u);
			}
		cv_ptr_copy_us->image /= max_inv_scale;
		us_img_pub_.publish(cv_ptr_copy_us->toImageMsg());
		ROS_INFO("max inv scale is %f", max_inv_scale);

		// upsampling method: s
		// cv::Mat mat_A = cv::Mat(3 * num_points, 2, CV_32FC1);
		// cv::Mat mat_b = cv::Mat(3 * num_points, 1, CV_32FC1);

		// for(int i = 0; i < num_points; i++)
		// {
		// 	// use the u, v channels
		// 	int tmp_u = int(orbmsg->channels[0].values[i] / ORB_WIDTH * MEGA_WIDTH);
		// 	int tmp_v = int(orbmsg->channels[1].values[i] / ORB_HEIGHT * MEGA_HEIGHT);
		// 	float tmp_depth = 1.f/cv_ptr->image.at<float>(tmp_v, tmp_u);

		// 	mat_A.at<float>(3*i, 0) = (tmp_u - U0) / Ku * tmp_depth;
		// 	mat_A.at<float>(3*i, 1) = 0.f;
		// 	mat_A.at<float>(3*i+1, 0) = (tmp_v - V0) / Kv * tmp_depth;
		// 	mat_A.at<float>(3*i+1, 1) = 0.f;
		// 	mat_A.at<float>(3*i+2, 0) = tmp_depth;
		// 	mat_A.at<float>(3*i+2, 1) = 1.f;

		// 	mat_b.at<float>(3*i, 0) = orbmsg->points[i].x;
		// 	mat_b.at<float>(3*i+1, 0) = orbmsg->points[i].y;
		// 	mat_b.at<float>(3*i+2, 0) = orbmsg->points[i].z;
		// }

		// cv::Mat mat_A_t = mat_A.t();
		// cv::Mat mat_A_t_mat_A = mat_A_t * mat_A;
		// cv::Mat unknown = mat_A_t_mat_A.inv() * mat_A_t * mat_b;
		
		// float scale = unknown.at<float>(0,0);
		// float offset = unknown.at<float>(1,0);

		// // ROS_INFO("unknown size: %d by %d", unknown.rows, unknown.cols);
		// ROS_INFO("scale: %f, offset: %f", scale, offset);

		sensor_msgs::PointCloud pclMsg2;

		for (size_t v = 0; v < cv_ptr->image.rows; v++)
			for (size_t u = 0; u < cv_ptr->image.cols; u++)
			{
				// ROS_INFO("u at %d, v at %d", u, v);
				geometry_msgs::Point32 pt;
				float ptz = exp(1.f/cv_ptr->image.at<float>(v,u));
				float scale = mat_scale.at<float>(v,u);
				if(scale < 0) {
					pt.x = 0;
					pt.y = 0;
					pt.z = 0;	
				} else {
					pt.x = (u - U0) / Ku * scale * ptz;
					pt.y = (v - V0) / Kv * scale * ptz;
					pt.z = ptz * scale;
				}
				pclMsg2.points.push_back(pt);
			}
		pclMsg2.header = cv_ptr->header;
		pclMsg2.header.frame_id = FRAME_ID_CAMERA;

		us_pcl_pub_.publish(pclMsg2);




		// // upsampling method (a,b)
		// cv::Mat mat_A = cv::Mat(3 * num_points, 2, CV_32FC1);
		// cv::Mat mat_b = cv::Mat(3 * num_points, 1, CV_32FC1);

		// for(int i = 0; i < num_points; i++)
		// {
		// 	// use the u, v channels
		// 	int tmp_u = int(orbmsg->channels[0].values[i] / ORB_WIDTH * MEGA_WIDTH);
		// 	int tmp_v = int(orbmsg->channels[1].values[i] / ORB_HEIGHT * MEGA_HEIGHT);
		// 	float tmp_depth = 1.f/cv_ptr->image.at<float>(tmp_v, tmp_u);

		// 	mat_A.at<float>(3*i, 0) = (tmp_u - U0) / Ku * tmp_depth;
		// 	mat_A.at<float>(3*i, 1) = 0.f;
		// 	mat_A.at<float>(3*i+1, 0) = (tmp_v - V0) / Kv * tmp_depth;
		// 	mat_A.at<float>(3*i+1, 1) = 0.f;
		// 	mat_A.at<float>(3*i+2, 0) = tmp_depth;
		// 	mat_A.at<float>(3*i+2, 1) = 1.f;

		// 	mat_b.at<float>(3*i, 0) = orbmsg->points[i].x;
		// 	mat_b.at<float>(3*i+1, 0) = orbmsg->points[i].y;
		// 	mat_b.at<float>(3*i+2, 0) = orbmsg->points[i].z;
		// }

		// cv::Mat mat_A_t = mat_A.t();
		// cv::Mat mat_A_t_mat_A = mat_A_t * mat_A;
		// cv::Mat unknown = mat_A_t_mat_A.inv() * mat_A_t * mat_b;
		
		// float scale = unknown.at<float>(0,0);
		// float offset = unknown.at<float>(1,0);

		// // ROS_INFO("unknown size: %d by %d", unknown.rows, unknown.cols);
		// ROS_INFO("scale: %f, offset: %f", scale, offset);

		// sensor_msgs::PointCloud pclMsg2;

		// for (size_t v = 0; v < cv_ptr->image.rows; v++)
		// 	for (size_t u = 0; u < cv_ptr->image.cols; u++)
		// 	{
		// 		// ROS_INFO("u at %d, v at %d", u, v);
		// 		geometry_msgs::Point32 pt;
		// 		float ptz = 1.f/cv_ptr->image.at<float>(v,u);
		// 		pt.x = (u - U0) / Ku * scale * ptz;
		// 		pt.y = (v - V0) / Kv * scale * ptz;
		// 		pt.z = ptz * scale + offset;
		// 		pclMsg2.points.push_back(pt);
		// 	}
		// pclMsg2.header = cv_ptr->header;
		// pclMsg2.header.frame_id = FRAME_ID_CAMERA;

		// us_pcl_pub_.publish(pclMsg2);
	}

	void orbCb(const sensor_msgs::PointCloudPtr& msg)
	{
		// pcl_mutex.lock();
		orbmsg = msg;
		// pcl_mutex.unlock();
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

	void draw_points(cv::Mat& img, vector<cv::Point3f>& points)
	{
		for (vector<cv::Point3f>::iterator it = points.begin(); it != points.end(); it++)
		{
			cv::Point3f tmp = *it;
			cv::circle(img, cv::Point(cvRound(tmp.x), cvRound(tmp.y)), 5, tmp.z, -1);
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

	void pt_neighbours_in_subdiv(int pt, cv::Subdiv2D& subdiv, set<int>& vec_neighbours)
	{
		// ROS_INFO("%d starts adding neighbours", pt);
		int edge;
		subdiv.getVertex(pt, &edge);
		if(pt != subdiv.edgeOrg(edge))
			edge = subdiv.rotateEdge(edge, 2);

		int neighbour_pt = subdiv.edgeDst(edge);
		while(vec_neighbours.find(neighbour_pt) == vec_neighbours.end())
		{
			vec_neighbours.insert(neighbour_pt);
			// ROS_INFO("%d has neighbour %d", pt, neighbour_pt);
			edge = subdiv.nextEdge(edge);
			neighbour_pt = subdiv.edgeDst(edge);
		}
		// ROS_INFO("# of vec_neighbours is %ld", vec_neighbours.size());
	}

	float look_up_pt_inv_mega_depth(cv::Point2f pt, cv::Mat mat_mega)
	{
		return exp(-1.f / mat_mega.at<float>(cvRound(pt.y), cvRound(pt.x)));
	}

	float look_up_pt_orb_depth(cv::Point2f pt, cv::Mat mat_orb)
	{
		return mat_orb.at<float>(cvRound(pt.y), cvRound(pt.x));
	}

	float look_up_pt_scale(cv::Point2f pt, cv::Mat mat_scale, float default_scale)
	{
		if(fabs(pt.x) >= MEGA_WIDTH || fabs(pt.y) >= MEGA_WIDTH)
			return default_scale;
		return mat_scale.at<float>(cvRound(pt.y), cvRound(pt.x));
	}

	float look_up_pt_scale(cv::Point2f pt, cv::Mat mat_orb, cv::Mat mat_mega, float default_scale)
	{
		if(fabs(pt.x) >= MEGA_WIDTH || fabs(pt.y) >= MEGA_WIDTH)
			return default_scale;

		float inv_mega_depth = look_up_pt_inv_mega_depth(pt, mat_mega);
		float orb_depth = look_up_pt_orb_depth(pt, mat_orb);

		// compute the scale: orb_depth / mega_depth
		return inv_mega_depth * orb_depth;
	}

	float squared_diff_in_mega_depth_from(float depth, cv::Point2f pt, cv::Mat mat_mega)
	{
		if(fabs(pt.x) >= MEGA_WIDTH || fabs(pt.y) >= MEGA_WIDTH)
			return (depth - exp(1.f)) * (depth - exp(1.f));	// assume the default mega_depth is 1, i.e., the nearest possible log-depth is 1 so depth is e

		float inv_mega_depth = look_up_pt_inv_mega_depth(pt, mat_mega);
		return (depth - 1.f/inv_mega_depth) * (depth - 1.f/inv_mega_depth);
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

	float compute_weighted_scale(float scale1, float scale2, float scale3, float squared_diff1, float squared_diff2, float squared_diff3)
	{
		bool check1 = (squared_diff1 > 3 * SIGMA);
		bool check2 = (squared_diff2 > 3 * SIGMA);
		bool check3 = (squared_diff3 > 3 * SIGMA);

		if (check1 && check2 && check3)
			return (scale1 + scale2 + scale3) / 3.f;

		if (check1 && check2)
			return scale3;
		if (check1 && check3)
			return scale2;
		if (check2 && check3)
			return scale1;

		float weight1 = exp(- squared_diff1 / SQUARED_SIGMA_R);
		float weight2 = exp(- squared_diff2 / SQUARED_SIGMA_R);
		float weight3 = exp(- squared_diff3 / SQUARED_SIGMA_R);

		if (check1)
			return (weight2 * scale2 + weight3 * scale3) / (weight2 + weight3);
		if (check2)
			return (weight1 * scale1 + weight3 * scale3) / (weight1 + weight3);
		if (check3)
			return (weight1 * scale1 + weight2 * scale2) / (weight1 + weight2);

		return (weight1 * scale1 + weight2 * scale2 + weight3 * scale3) / (weight1 + weight2 + weight3);
	}
};


int main(int argc, char **argv)
{
	ros::init(argc, argv, "upsample_node");

	UpSampler us;

	ros::NodeHandle nh_;
	us.mega_sub_ = nh_.subscribe("/mega/inv_depth", 1, &UpSampler::imageCb, &us);
	us.orb_sub_ = nh_.subscribe("/orb_slam/pcl_cam", 1, &UpSampler::orbCb, &us);

	us.mega_pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud>("/us/mega_raw", 1);
	us.orb_pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud>("/us/orb_raw", 1);
	us.us_pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud>("/us/pcl", 1);
	us.tri_img_pub_ = nh_.advertise<sensor_msgs::Image>("/us/triangulate_raw", 1);
	us.scale_img_pub_ = nh_.advertise<sensor_msgs::Image>("/us/scale_raw", 1);
	us.us_img_pub_ = nh_.advertise<sensor_msgs::Image>("/us/img", 1);

	ros::spin();
	return 0;
}
