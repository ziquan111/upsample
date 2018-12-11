#include <ros/ros.h>
#include <std_msgs/String.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <geometry_msgs/Point32.h>
#include <sensor_msgs/PointCloud.h>

#include <iostream>
#include <sstream>
// #include <mutex>
#include <time.h>

#include <algorithm>
#include <numeric>
#include <map>

#include <sys/types.h>
#include <dirent.h>

#define Ku 338.7884
#define Kv 444.1741
#define U0 256.0
#define V0 192.0

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

#define DO_BLUR false

// std::mutex pcl_mutex;

using namespace std;
// using namespace cv;
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

	void processImage(string file_name)
	{
		cv::Mat img;
		img = cv::imread(file_name, cv::IMREAD_COLOR);

		if ( img.empty() )
		{
			ROS_ERROR("Could not open for find the image");
			return;
		}

		cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
		cv::imshow("Display", img);

		cv::waitKey(0);
		// cv_bridge::CvImage img_bridge;
		// sensor_msgs::Image img_msg;

		// std_msgs::Header header;
		// header.seq = counter;
		// header.stamp = ros::Time::now();

		// img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, img);
		// img_bridge.toImageMsg(img_msg);
	}

	void imageCb(const sensor_msgs::ImageConstPtr& msg)
	{
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

		sensor_msgs::PointCloud pclMsg;

		for (size_t v = 0; v < cv_ptr->image.rows; v++)
			for (size_t u = 0; u < cv_ptr->image.cols; u++)
			{
				geometry_msgs::Point32 pt;
				pt.z = exp(1.f / cv_ptr->image.at<float>(v,u));	// x and y depend on z
				pt.x = (u - U0) / Ku * pt.z;
				pt.y = (v - V0) / Kv * pt.z;
				pclMsg.points.push_back(pt);
			}
		pclMsg.header = cv_ptr->header;
		pclMsg.header.frame_id = FRAME_ID_CAMERA;

		mega_pcl_pub_.publish(pclMsg);

		// 
		// repub orb
		// 

		if(orbmsg == NULL)
			return;
		int num_points = orbmsg->points.size();
		ROS_INFO("<%u> processing orb message---, %d", msg->header.seq, num_points);
		orb_pcl_pub_.publish(orbmsg);

		// 
		// upsampling using triangluate approach
		// 

		// STEP 1: remove outlier

		// perform the delaunary //, estimate a (weighted) average scale, and prepare smoothing patches
		// prepare for delaunary
		
		clock_t timer = clock();
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
			float tmp_orb_depth = orbmsg->points[i].z;
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
		
		float time_used_to_locate = 0.f;
		float time_used_to_prepare = 0.f;
		float time_used_to_compute = 0.f;
		//clock_t timer;

		float scale_max = -1.f, scale_min = 9999.f;
		float scale_sum = 0.f;
		float scale_count = 0.f;
		cv::Mat mat_log_scale(MEGA_HEIGHT, MEGA_WIDTH, CV_32FC1, -1); // default scale is set to be -1
#pragma omp parallel
#pragma omp for	
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

// 		map<int, vector<int> > dic_nearby_pt_id; // map from edge id to a set of vertex ids
// 		vector<vector<vector<int> > > vv_vector_nearby_pt_id;
// 		vv_vector_nearby_pt_id.resize(cv_ptr->image.rows);
// 		// vector<int> vec_nearby_pt_id;
// 		for (size_t v = 0; v < cv_ptr->image.rows; v++)
// 		{
// 			vv_vector_nearby_pt_id[v].resize(cv_ptr->image.cols);
// 			for (size_t u = 0; u < cv_ptr->image.cols; u++)
// 			{
// 				cv::Point2f pt(u, v);
// 				// float log_mega_depth = 1.f / look_up_pt_inv_log_mega_depth(pt, cv_ptr->image);

// 				timer = clock();
// 				// vector<int> vec_nearby_pt_id;
// 				// vec_nearby_pt_id.clear();
// 				int flag = locate_pt_in_subdiv2(pt, subdiv_clean, vv_vector_nearby_pt_id[v][u], dic_nearby_pt_id);
// 				// locate_pt_in_subdiv2(pt, subdiv_clean, dic_nearby_pt_id);
// 				timer = clock() - timer;
// 				time_used_to_locate += ((float)timer)/CLOCKS_PER_SEC;
// 			}
// 		}
// 		vector<vector<vector<float> > > vv_vec_nearby_log_scale, vv_vec_nearby_squared_diff;
// 		vv_vec_nearby_log_scale.resize(cv_ptr->image.rows);
// 		vv_vec_nearby_squared_diff.resize(cv_ptr->image.rows);
// // #pragma omp parallel
// // #pragma omp for	
// 		for (size_t v = 0; v < cv_ptr->image.rows; v++)
// 		{
// 			vv_vec_nearby_log_scale[v].resize(cv_ptr->image.cols);
// 			vv_vec_nearby_squared_diff[v].resize(cv_ptr->image.cols);
// 			for (size_t u = 0; u < cv_ptr->image.cols; u++)
// 			{
// 				cv::Point2f pt(u, v);
// 				float log_mega_depth = 1.f / look_up_pt_inv_log_mega_depth(pt, cv_ptr->image);

// 				timer = clock();
// 				// vector<float> vec_nearby_log_scale, vec_nearby_squared_diff;
// 				// vec_nearby_log_scale.clear();
// 				// vec_nearby_squared_diff.clear();
// 				// for (vector<int>::iterator it = vec_nearby_pt_id.begin(); it != vec_nearby_pt_id.end(); it++)
// 				vector<int> very_tmp = vv_vector_nearby_pt_id[v][u];
// 				for (vector<int>::iterator it = very_tmp.begin(); it != very_tmp.end(); it++)
// 				{	
// 					// vec_nearby_log_scale.push_back(vec_log_scale_clean[*it]);
// 					vv_vec_nearby_log_scale[v][u].push_back(vec_log_scale_clean[*it]);
// 					float tmp_diff = squared_diff_in_log_mega_depth_from(log_mega_depth, subdiv_clean.getVertex(*it), cv_ptr->image);
// 					// vec_nearby_squared_diff.push_back(tmp_diff);
// 					vv_vec_nearby_squared_diff[v][u].push_back(tmp_diff);
// 				}
// 				timer = clock() - timer;
// 				time_used_to_prepare += ((float)timer)/CLOCKS_PER_SEC;

// 				timer = clock();
// 				// mat_log_scale.at<float>(v, u) = compute_weighted_log_scale(vec_nearby_log_scale, vec_nearby_squared_diff);
// 				mat_log_scale.at<float>(v, u) = compute_weighted_log_scale(vv_vec_nearby_log_scale[v][u], vv_vec_nearby_squared_diff[v][u]);
// 				timer = clock() - timer;
// 				time_used_to_compute += ((float)timer)/CLOCKS_PER_SEC;
// 			}
// 		}
		
		// ROS_INFO("<%u> time to locate: %f, time to prepare: %f, time to compute: %f", msg->header.seq, time_used_to_locate, time_used_to_prepare, time_used_to_compute);
		// if (DO_BLUR)
		// 	cv::GaussianBlur(mat_scale, mat_scale, cv::Size(0,0), 10);
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(mat_log_scale, &minVal, &maxVal, &minLoc, &maxLoc);
		ROS_INFO("<%u> max scale is %lf, min scale is %lf",  msg->header.seq, exp(maxVal), exp(minVal));
		//ROS_INFO("time to locate %f, time to compute %f", time_used_locate, time_used_compute);

		// visualize triangulation
		vector<cv::Point3f> points_to_draw, neighbour_points_to_draw;
		float max_log_scale_clean = max_in_vector(vec_log_scale_clean);
		for (int i = 0; i < num_unique_points_clean; i++)
		{
			cv::Point2f tmp_pt = subdiv_clean.getVertex(i) ;
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
		ROS_INFO("max log scale is %f", max_log_scale);

		// visualize the upsampled output in 3D
		sensor_msgs::PointCloud pclMsg2;

		for (size_t v = 0; v < cv_ptr->image.rows; v++)
			for (size_t u = 0; u < cv_ptr->image.cols; u++)
			{
				// ROS_INFO("u at %d, v at %d", u, v);
				geometry_msgs::Point32 pt;
				// float ptz = exp(1.f / cv_ptr->image.at<float>(v,u));
				float scale = exp(mat_log_scale.at<float>(v,u));
				float scaled_ptz = exp(mat_log_scale.at<float>(v,u) + 1.f / cv_ptr->image.at<float>(v,u));
				if(scale < 0) {
					pt.x = 0;
					pt.y = 0;
					pt.z = 0;	
				} else {
					pt.x = (u - U0) / Ku * scaled_ptz;
					pt.y = (v - V0) / Kv * scaled_ptz;
					pt.z = scaled_ptz;
				}
				pclMsg2.points.push_back(pt);
			}
		pclMsg2.header = cv_ptr->header;
		pclMsg2.header.frame_id = FRAME_ID_CAMERA;

		us_pcl_pub_.publish(pclMsg2);
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

	// float look_up_pt_inv_mega_depth(cv::Point2f pt, cv::Mat mat_mega)
	// {
	// 	return exp(-1.f / mat_mega.at<float>(cvRound(pt.y), cvRound(pt.x)));
	// }

	// float look_up_pt_orb_depth(cv::Point2f pt, cv::Mat mat_orb)
	// {
	// 	return mat_orb.at<float>(cvRound(pt.y), cvRound(pt.x));
	// }

	// float look_up_pt_scale(cv::Point2f pt, cv::Mat mat_scale, float default_scale)
	// {
	// 	if(fabs(pt.x) >= MEGA_WIDTH || fabs(pt.y) >= MEGA_WIDTH)
	// 		return default_scale;
	// 	return mat_scale.at<float>(cvRound(pt.y), cvRound(pt.x));
	// }

	// float look_up_pt_scale(cv::Point2f pt, cv::Mat mat_orb, cv::Mat mat_mega, float default_scale)
	// {
	// 	if(fabs(pt.x) >= MEGA_WIDTH || fabs(pt.y) >= MEGA_WIDTH)
	// 		return default_scale;

	// 	float inv_mega_depth = look_up_pt_inv_mega_depth(pt, mat_mega);
	// 	float orb_depth = look_up_pt_orb_depth(pt, mat_orb);

	// 	// compute the scale: orb_depth / mega_depth
	// 	return inv_mega_depth * orb_depth;
	// }


	float squared_diff_in_log_mega_depth_from(float log_depth, cv::Point2f pt, cv::Mat mat_mega)
	{
		if(fabs(pt.x) >= MEGA_WIDTH || fabs(pt.y) >= MEGA_WIDTH)
			return (log_depth - 1.f) * (log_depth - 1.f);	// assume the default mega_depth is 1, i.e., the nearest possible

		float log_mega_depth = 1.f / look_up_pt_inv_log_mega_depth(pt, mat_mega);
		return (log_depth - log_mega_depth) * (log_depth - log_mega_depth);
	}

	// float squared_diff_in_mega_depth_from(float depth, cv::Point2f pt, cv::Mat mat_mega)
	// {
	// 	if(fabs(pt.x) >= MEGA_WIDTH || fabs(pt.y) >= MEGA_WIDTH)
	// 		return (depth - exp(1.f)) * (depth - exp(1.f));	// assume the default mega_depth is 1, i.e., the nearest possible log-depth is 1 so depth is e

	// 	float inv_mega_depth = look_up_pt_inv_mega_depth(pt, mat_mega);
	// 	return (depth - 1.f/inv_mega_depth) * (depth - 1.f/inv_mega_depth);
	// }

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

	// float compute_weighted_scale(float scale1, float scale2, float scale3, float squared_diff1, float squared_diff2, float squared_diff3)
	// {
	// 	bool check1 = (squared_diff1 > 3 * SIGMA);
	// 	bool check2 = (squared_diff2 > 3 * SIGMA);
	// 	bool check3 = (squared_diff3 > 3 * SIGMA);

	// 	if (check1 && check2 && check3)
	// 		return (scale1 + scale2 + scale3) / 3.f;

	// 	if (check1 && check2)
	// 		return scale3;
	// 	if (check1 && check3)
	// 		return scale2;
	// 	if (check2 && check3)
	// 		return scale1;

	// 	float weight1 = exp(- squared_diff1 / SQUARED_SIGMA_R);
	// 	float weight2 = exp(- squared_diff2 / SQUARED_SIGMA_R);
	// 	float weight3 = exp(- squared_diff3 / SQUARED_SIGMA_R);

	// 	if (check1)
	// 		return (weight2 * scale2 + weight3 * scale3) / (weight2 + weight3);
	// 	if (check2)
	// 		return (weight1 * scale1 + weight3 * scale3) / (weight1 + weight3);
	// 	if (check3)
	// 		return (weight1 * scale1 + weight2 * scale2) / (weight1 + weight2);

	// 	return (weight1 * scale1 + weight2 * scale2 + weight3 * scale3) / (weight1 + weight2 + weight3);
	// }

	void union_of_sets(set<int>& input, set<int>& output)
	{
		for(set<int>::iterator it = input.begin(); it != input.end(); it++)
		{
			output.insert(*it);
		}
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

	string img_folder_name = "/home/ziquan/Downloads/Make3D/400Img/";
	string depth_folder_name = "/home/ziquan/Downloads/Make3D/400Depth/";
	DIR* img_dirp = opendir(img_folder_name.c_str());
	DIR* depth_dirp = opendir(depth_folder_name.c_str());

	int is_open = false;
	struct dirent * dp;
	while ((dp = readdir(img_dirp)) != NULL)
	{
		// stringstream ss_img_file_name, ss_depth_file_name;
		// ss_img_file_name << dp->d_name;
		string img_file_name = dp->d_name;
		cout << img_file_name << endl;
		if (!is_open)
		{
			if (img_file_name.length() <= 2)
				continue;
			stringstream ss;
			ss << img_folder_name << img_file_name;
			us.processImage(ss.str());
			is_open = true;
		}
		// strcpy(dp->d_name, img_file_name.c_str());// = ss_img_file_name.str(dp->d_name);
	
		// ss_depth_file_name << "depth_sph_corr" << img_file_name.substr(3);
		// string depth_file_name = ss_depth_file_name.str();
		// ROS_INFO("%d, %s", counter++, img_file_name);//, depth_file_name);
	}
	closedir(img_dirp);

	// counter = 0;
	// while ((dp = readdir(depth_dirp)) != NULL)
	// {
	// 	ROS_INFO("%d, %s", counter++,dp->d_name);
	// }
	// closedir(depth_dirp);

	// string file_name = "/home/ziquan/Downloads/Make3D/400Img/img-10.21op2-p-046t000.jpg";
	// us.processImage(file_name);

	ros::spin();
	return 0;
}
