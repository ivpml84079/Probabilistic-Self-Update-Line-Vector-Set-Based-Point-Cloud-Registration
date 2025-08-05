// An example showing TEASER++ registration with the Stanford bunny model
#include <chrono>
#include <iostream>
#include <random>

#include <Eigen/Core>

#include <teaser/ply_io.h>
#include <teaser/registration.h>

//CHANGE NEW INCLUDE 
#include <fstream>
#include <streambuf>
#include <unistd.h> 
#include <cmath>
#include <filesystem>
#include <string>

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

namespace fs = std::filesystem;

// Macro constants for generating noise and outliers
#define NOISE_BOUND 0.05
// #define N_OUTLIERS_RATE 0.9
double N_OUTLIERS_RATE[5] = {0.5, 0.6, 0.7, 0.8, 0.9};
std::string N_OUTLIERS_RATE_str[5] = {"0.5", "0.6", "0.7", "0.8", "0.9"};
int N_index = 0;
#define PI 3.1415926

int unknownScale = 0;

inline double getAngularError(Eigen::Matrix3d R_exp, Eigen::Matrix3d R_est) {
  return std::abs(std::acos(fmin(fmax(((R_exp.transpose() * R_est).trace() - 1) / 2, -1.0), 1.0)))*180/PI;
}

void compute_normal(Eigen::Matrix<double, 3, Eigen::Dynamic>& src, 
                    Eigen::Matrix<double, 3, Eigen::Dynamic>& tgt, 
                    Eigen::Matrix<double, 3, Eigen::Dynamic>& src_normals, 
                    Eigen::Matrix<double, 3, Eigen::Dynamic>& tgt_normals){
    pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    src_cloud->width = src.cols();
    src_cloud->height = 1;
    src_cloud->is_dense = false;
    src_cloud->points.resize(src_cloud->width * src_cloud->height);
    tgt_cloud->width = tgt.cols();
    tgt_cloud->height = 1;
    tgt_cloud->is_dense = false;
    tgt_cloud->points.resize(tgt_cloud->width * tgt_cloud->height);

    for (size_t i = 0; i < src_cloud->points.size(); ++i)
    {
        src_cloud->points[i].x = src(0, i);
        src_cloud->points[i].y = src(1, i);
        src_cloud->points[i].z = src(2, i);
    }
    for (size_t i = 0; i < tgt_cloud->points.size(); ++i)
    {
        tgt_cloud->points[i].x = tgt(0, i);
        tgt_cloud->points[i].y = tgt(1, i);
        tgt_cloud->points[i].z = tgt(2, i);
    }
    pcl::PointCloud<pcl::Normal>::Ptr src_normals_cld(new pcl::PointCloud<pcl::Normal>());
    pcl::PointCloud<pcl::Normal>::Ptr tgt_normals_cld(new pcl::PointCloud<pcl::Normal>());

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    ne.setSearchMethod(tree);
    // ne.setRadiusSearch(0.15); //KITTI
    ne.setKSearch(20); //3DMATCH
    ne.setInputCloud(src_cloud);
    ne.compute(*src_normals_cld);

    ne.setInputCloud(tgt_cloud);
    ne.compute(*tgt_normals_cld);

    for (size_t i = 0; i < src_normals_cld->points.size(); ++i)
    {
        src_normals(0, i) = src_normals_cld->points[i].normal_x;
        src_normals(1, i) = src_normals_cld->points[i].normal_y;
        src_normals(2, i) = src_normals_cld->points[i].normal_z;
    }
    for (size_t i = 0; i < tgt_normals_cld->points.size(); ++i)
    {
        tgt_normals(0, i) = tgt_normals_cld->points[i].normal_x;
        tgt_normals(1, i) = tgt_normals_cld->points[i].normal_y;
        tgt_normals(2, i) = tgt_normals_cld->points[i].normal_z;
    }
}

int histogram_outlier_removal(Eigen::Matrix<double, 3, Eigen::Dynamic>& src_normals, 
                              Eigen::Matrix<double, 3, Eigen::Dynamic>& tgt_normals, 
                              std::vector<int>& keep_mask) {
  std::vector<double> all_angles(src_normals.cols(), -1);
  std::vector<double> remain_angles;
  double o_max = 0, o_min = INT_MAX;
  double angle_sum = 0;

  for(int i = 0; i < src_normals.cols(); ++i) {
    Eigen::Vector3d vec1 = src_normals.col(i).normalized();
    Eigen::Vector3d vec2 = tgt_normals.col(i).normalized();

    double cos_theta = vec1.dot(vec2);
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));

    double angle_deg = std::acos(cos_theta) * 180.0 / M_PI;
    if(std::isnan(angle_deg)) continue;

    remain_angles.push_back(angle_deg);
    all_angles[i] = angle_deg;
    o_min = std::min(angle_deg, o_min);
    o_max = std::max(angle_deg, o_max);
    angle_sum += angle_deg;
  }

  double angle_mean = angle_sum / remain_angles.size();
  double angle_squaredSum = 0.0;
  for (const double& deg : remain_angles) {
    angle_squaredSum += std::pow(deg - angle_mean, 2);
  }
  double angle_variance = angle_squaredSum / remain_angles.size();
  double angle_standard_deviation = std::sqrt(angle_variance);

  double o_bin_width = 3.49 * angle_standard_deviation / std::pow(remain_angles.size(), (1.0/3.0));

  int o_histogram_size = std::max(1, (int) std::ceil((o_max - o_min) / o_bin_width));

  std::vector<std::vector<int>> angles_histogram(o_histogram_size, std::vector<int>(0));
  int remain_count = 0;
  float stdDevMultiplier = 0.5;

  int peak_id = 0;
  int peak_height = 0;
  
  std::vector<int> o_histogram_height(o_histogram_size, 0);
  for(int i = 0; i < src_normals.cols(); i++) {
    if(all_angles[i] == -1) continue;
    int bin_idx = (all_angles[i] - o_min) / o_bin_width;
    angles_histogram[bin_idx].push_back(i);
    o_histogram_height[bin_idx]++;
    if(angles_histogram[bin_idx].size() > peak_height){
      peak_height = angles_histogram[bin_idx].size();
      peak_id = bin_idx;
    }
  }

  // for(int i = 0; i < o_histogram_height.size(); i++) std::cout << o_histogram_height[i] << std::endl;

  double o_height_mean = std::accumulate(o_histogram_height.begin(), o_histogram_height.end(), 0.0) / o_histogram_height.size();
  double o_height_variance = 0.0;
  for (int height : o_histogram_height) {
    o_height_variance += std::pow(height - o_height_mean, 2);
  }
  double o_height_stdDev = std::sqrt(o_height_variance / o_histogram_height.size());
  double o_height_threshold = o_height_mean + stdDevMultiplier * o_height_stdDev;

  // std::cout << "o_height_threshold: " << o_height_threshold << std::endl;

  for(int i = 0; i < angles_histogram.size(); i++) {
    // if(std::abs(i - peak_id) > 2){
    //   for(int j = 0; j < angles_histogram[i].size(); j++) {
    //     keep_mask[angles_histogram[i][j]] = -1;
    //   }
    // }
    if(angles_histogram[i].size() > o_height_threshold) {
      for(int j = 0; j < angles_histogram[i].size(); j++) {
        keep_mask[angles_histogram[i][j]] = 1;
        remain_count++;
      }
    }
  }

  // std::cout << "remain_count: " << remain_count << std::endl;
  return remain_count;
}

void mask_filter( Eigen::Matrix<double, 3, Eigen::Dynamic>& src, 
                  Eigen::Matrix<double, 3, Eigen::Dynamic>& tgt, 
                  Eigen::Matrix<double, 3, Eigen::Dynamic>& src_reduce, 
                  Eigen::Matrix<double, 3, Eigen::Dynamic>& tgt_reduce, 
                  std::vector<int> keep_mask,
                  std::map<int, int>& reduce_map){
  int colIndex = 0;
  for(int i = 0; i < src.cols(); ++i) {
    if(keep_mask[i] == 1){
      reduce_map[i] = colIndex;
      src_reduce.col(colIndex) = src.col(i);
      tgt_reduce.col(colIndex) = tgt.col(i);
      colIndex++;
    }
  }
}

void addNoiseAndOutliers(Eigen::Matrix<double, 3, Eigen::Dynamic>& tgt) {
  Eigen::Matrix<double, 3, Eigen::Dynamic> noise =
      Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, tgt.cols()) * NOISE_BOUND;
  NOISE_BOUND / 2;
  tgt = tgt + noise;

  // Add outliers
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis2(0, tgt.cols() - 1); // pos of outliers
  std::uniform_real_distribution<> dis3_lb(-10.0, -5.0); //-10~-5, 5~10
  std::uniform_real_distribution<> dis3_ub(5.0, 10.0);
  std::uniform_real_distribution<> disC(0.0, 1.0);
  std::vector<bool> expected_outlier_mask(tgt.cols(), false);
  int outliers = tgt.cols() * N_OUTLIERS_RATE[N_index];
  for (int i = 0; i < outliers; ++i) {
    int c_outlier_idx = dis2(gen);
    assert(c_outlier_idx < expected_outlier_mask.size());
	if(expected_outlier_mask[c_outlier_idx])
	{
		i--;
		continue;
	}
    expected_outlier_mask[c_outlier_idx] = true;
    Eigen::Matrix<double, 3, 1> Tt;
    for(int tt = 0; tt < 3; tt++)
    {
      if(disC(gen) <= 0.5)
        Tt(tt) = dis3_lb(gen);
      else
        Tt(tt) = dis3_ub(gen);
    }
    tgt.col(c_outlier_idx) += Tt; // random translation
  }
}

int main(int argv, char* argc[]) {  
  const std::string folder_path = "./dataset/artificial";

  if (!fs::exists(folder_path)) {
    std::cerr << folder_path << std::endl;
    return -1;
  }
  // 確保 Result 資料夾存在
  std::string result_base_path = "./Result/artificial";
  if (!fs::exists(result_base_path)) {
      fs::create_directory(result_base_path);
  }
  std::ofstream oFile;
  oFile.open(result_base_path + "/result.csv", std::ios::out | std::ios::trunc);
  for(N_index = 4;N_index >= 0; N_index--){
    double sumSE = 0, sumRE = 0, sumTE = 0, sumTime = 0, sumRMSE = 0;
    int totalData = 0;
    for(const auto& entry : fs::directory_iterator(folder_path)){	
      totalData++;
      teaser::PLYReader reader;
      teaser::PointCloud src_cloud;
      auto status = reader.read(entry.path().string(), src_cloud);
      std::cout << entry.path().string() << std::endl;
      // for(int t = 1; t < argv; t++)
      // {
        // Load the .ply file
        // teaser::PLYReader reader;
        // teaser::PointCloud src_cloud;
        // auto status = reader.read(std::string(argc[t]), src_cloud);
        int N = src_cloud.size();
        // Convert the point cloud to Eigen
        Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, N);
        for (size_t i = 0; i < N; ++i) {
          src.col(i) << src_cloud[i].x, src_cloud[i].y, src_cloud[i].z;
        }      

        // Homogeneous coordinates
        Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
        src_h.resize(4, src.cols());
        src_h.topRows(3) = src;
        src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(N);

        srand(20241127);
        //test mutiple + excel
        double scale_error_sum = 0, scale_error_sum_of_square = 0;
        double angle_error_sum = 0, angle_error_sum_of_square = 0;
        double trans_error_sum = 0, trans_error_sum_of_square = 0;
        double time_sum = 0, time_sum_of_square = 0;
        double RMSE_sum = 0, RMSE_sum_of_square = 0;
        double scale_best = 10000;
        double angle_best = 10000;
        double trans_best = 10000;
        double time_best = 10000;
        double RMSE_best = 10000;
        int testTime = 1;
        for(int i = 0; i < testTime; i++)
        {      
          //random
          double testScale = 1;
          if (unknownScale)
            testScale = 1 + 4 * ((1.0 - 0.0) * rand() / (RAND_MAX + 1.0) + 0.0);
          Eigen::Matrix4d T;
          T << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1;
          Eigen::Matrix<double, 3, Eigen::Dynamic> axis = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1);
          axis = axis/axis.norm();
          double randomAngle = (3.1416 - 0.0) * rand() / (RAND_MAX + 1.0) + 0.0;
          Eigen::Matrix<double, 3, Eigen::Dynamic> aa = randomAngle * axis;
          Eigen::Matrix<double, 3, 3> R;
          R << 1, 0, 0,
              0, 1, 0,
              0, 0, 1;
          if(aa.norm() >= 2e-16)
          {
            Eigen::Matrix<double, 3, Eigen::Dynamic> K = aa / aa.norm();
            Eigen::Matrix<double, 3, 3> K1;
            K1 << 0, -K(2), K(1),
                  K(2), 0, -K(0),
                  -K(1), K(0), 0;
            R = R + sin(aa.norm()) * K1 + (1 - cos(aa.norm())) * K1 * K1;
          }
          T.topLeftCorner(3, 3) = R;
          Eigen::Matrix<double, 3, 1> Tt;
          Tt(0) = (1.0 - 0.0) * rand() / (RAND_MAX + 1.0) + 0.0 - 0.5;
          Tt(1) = (1.0 - 0.0) * rand() / (RAND_MAX + 1.0) + 0.0 - 0.5;
          Tt(2) = (1.0 - 0.0) * rand() / (RAND_MAX + 1.0) + 0.0 - 0.5;
          Tt = 3 * ((1.0 - 0.0) * rand() / (RAND_MAX + 1.0) + 0.0) * Tt / Tt.norm();
          T.topRightCorner(3, 1) = Tt;

          // Apply transformation
          Eigen::Matrix<double, 4, Eigen::Dynamic> tgt_h = testScale * T * src_h;
          Eigen::Matrix<double, 3, Eigen::Dynamic> tgt = tgt_h.topRows(3);
          Eigen::Matrix<double, 3, Eigen::Dynamic> src_gt = tgt_h.topRows(3);
          // Add some noise & outliers
          addNoiseAndOutliers(tgt);
          // Run our modified TEASER++ registration
          // Prepare solver parameters
          teaser::RobustRegistrationSolver::Params params;
          params.noise_bound = NOISE_BOUND;
          params.cbar2 = 1;
          params.estimate_scaling = unknownScale;
          params.rotation_max_iterations = 100;
          params.rotation_gnc_factor = 1.4;
          params.rotation_estimation_algorithm =
          teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
          params.rotation_cost_threshold = 0.005;
          // Solve with our modified TEASER++ method with the propose C-RANSAC structure
          Eigen::Matrix<double, 3, Eigen::Dynamic> src_normals(3, src.cols());
          Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_normals(3, tgt.cols());
          src_normals.setZero();
          tgt_normals.setZero();
          compute_normal(src, tgt, src_normals, tgt_normals);
          std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
          std::map<int, int> reduce_map;
          std::vector<int> keep_mask(src.cols(), 0);
          int remain_count = histogram_outlier_removal(src_normals, tgt_normals, keep_mask);
          Eigen::Matrix<double, 3, Eigen::Dynamic> src_reduce(3, remain_count);
          Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_reduce(3, remain_count);
          src_reduce.setZero();
          tgt_reduce.setZero();
          mask_filter(src, tgt, src_reduce, tgt_reduce, keep_mask, reduce_map);
          params.ori_src.resize(3, src.cols());
          params.ori_dst.resize(3, tgt.cols());
          params.ori_src = src.topRows(3);
          params.ori_dst = tgt.topRows(3);
          params.keep_mask = keep_mask;
          params.reduce_map = reduce_map;

          teaser::RobustRegistrationSolver solver(params);
          solver.solve(src_reduce, tgt_reduce);
          std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

          auto solution = solver.getSolution();
          // if(fabs(testScale - solution.scale) > 0.1)
          // {
          //   i--;
          //   continue;
          // }
          double angleError = getAngularError(T.topLeftCorner(3, 3), solution.rotation);
          scale_error_sum += testScale - solution.scale;
          scale_error_sum_of_square += pow(testScale - solution.scale, 2);
          angle_error_sum += angleError;
          angle_error_sum_of_square += pow(angleError, 2);
          trans_error_sum += (T.topRightCorner(3, 1) - solution.translation).norm();
          trans_error_sum_of_square += pow((T.topRightCorner(3, 1) - solution.translation).norm(), 2);
          time_sum += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
          time_sum_of_square += pow(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0, 2);
          // Homogeneous coordinates
          Eigen::Matrix4d TRANSFORM;
          TRANSFORM <<  1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1;
          TRANSFORM.topLeftCorner(3, 3) = solution.rotation;
          TRANSFORM.topRightCorner(3, 1) = solution.translation;
          Eigen::Matrix<double, 3, Eigen::Dynamic> src_solve = (solution.scale * TRANSFORM * src_h).topRows(3);
          double residualSum = 0.0;
          for(int m = 0; m < src_solve.cols(); m++)
          {
            residualSum += pow((src_gt.col(m) - src_solve.col(m)).norm(), 2);
          }
          residualSum = sqrt(residualSum / src_solve.cols());
          RMSE_sum += residualSum;
          RMSE_sum_of_square += pow(residualSum, 2);

          // Compare results
          std::cout << "\n\n";
          std::cout << "=====================================" << std::endl;
          std::cout << "          C-RANSAC Results           " << std::endl;
          std::cout << "=====================================" << std::endl;
          std::cout << "Expected scale: " << std::endl;
          std::cout << testScale << std::endl;
          std::cout << "Estimated scale: " << std::endl;
          std::cout << solution.scale << std::endl;
          std::cout << "Error (): " << testScale - solution.scale
                    << std::endl;
          std::cout << std::endl;
          std::cout << "Expected rotation: " << std::endl;
          std::cout << T.topLeftCorner(3, 3) << std::endl;
          std::cout << "Estimated rotation: " << std::endl;
          std::cout << solution.rotation << std::endl;
          std::cout << "Error (deg): " << angleError
                    << std::endl;
          std::cout << std::endl;
          std::cout << "Expected translation: " << std::endl;
          std::cout << T.topRightCorner(3, 1) << std::endl;
          std::cout << "Estimated translation: " << std::endl;
          std::cout << solution.translation << std::endl;
          std::cout << "Error (m): " << (T.topRightCorner(3, 1) - solution.translation).norm() << std::endl;
          std::cout << std::endl;
          std::cout << "RMSE: " << residualSum << std::endl;
          std::cout << std::endl;
          std::cout << "Number of correspondences: " << N << std::endl;
          std::cout << "current outlier rate: " << N_OUTLIERS_RATE_str[N_index] << std::endl;
          std::cout << "Time taken (s): "
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() /
                          1000000.0
                    << std::endl;
          // std::cout << std::string(argc[t]) << " " << i << std::endl;
          if(residualSum < RMSE_best){
            scale_best = testScale - solution.scale;
            angle_best = angleError;
            trans_best = (T.topRightCorner(3, 1) - solution.translation).norm();
            time_best = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
            RMSE_best = residualSum;
          }
          // sleep(3);
        }
        // double scale_error_standard_deviation = sqrt(scale_error_sum_of_square / testTime - pow(scale_error_sum / testTime, 2)); //calculate standard deviation
        // double angle_error_standard_deviation = sqrt(angle_error_sum_of_square / testTime - pow(angle_error_sum / testTime, 2)); //calculate standard deviation
        // double trans_error_standard_deviation = sqrt(trans_error_sum_of_square / testTime - pow(trans_error_sum / testTime, 2)); //calculate standard deviation
        // double time_standard_deviation = sqrt(time_sum_of_square / testTime - pow(time_sum / testTime, 2)); //calculate standard deviation
        // double RMSE_standard_deviation = sqrt(RMSE_sum_of_square / testTime - pow(RMSE_sum / testTime, 2)); //calculate standard deviation
        
        std::cout << "=====================================" << std::endl;
        std::cout << "                Final                " << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "Scale_Error:, (esti. / grou.)" << std::endl;
        // std::cout << scale_error_sum / testTime + 2 * scale_error_standard_deviation << std::endl;
        // std::cout << scale_error_sum / testTime + 1 * scale_error_standard_deviation << std::endl;
        std::cout << scale_error_sum / testTime << std::endl;
        // std::cout << scale_error_sum / testTime - 1 * scale_error_standard_deviation << std::endl;
        // std::cout << scale_error_sum / testTime - 2 * scale_error_standard_deviation << std::endl;
        std::cout << std::endl << "Angle_Error:, (degree)" << std::endl;
        // std::cout << angle_error_sum / testTime + 2 * angle_error_standard_deviation << std::endl;
        // std::cout << angle_error_sum / testTime + 1 * angle_error_standard_deviation << std::endl;
        std::cout << angle_error_sum / testTime << std::endl;
        // std::cout << angle_error_sum / testTime - 1 * angle_error_standard_deviation << std::endl;
        // std::cout << angle_error_sum / testTime - 2 * angle_error_standard_deviation << std::endl;
        std::cout << std::endl << "Trans_Error:, (meter)" << std::endl;
        // std::cout << trans_error_sum / testTime + 2 * trans_error_standard_deviation << std::endl;
        // std::cout << trans_error_sum / testTime + 1 * trans_error_standard_deviation << std::endl;
        std::cout << trans_error_sum / testTime << std::endl;
        // std::cout << trans_error_sum / testTime - 1 * trans_error_standard_deviation << std::endl;
        // std::cout << trans_error_sum / testTime - 2 * trans_error_standard_deviation << std::endl;
        std::cout << std::endl << "Time:, (second)" << std::endl;
        // std::cout << time_sum / testTime + 2 * time_standard_deviation << std::endl;
        // std::cout << time_sum / testTime + 1 * time_standard_deviation << std::endl;
        std::cout << time_sum / testTime << std::endl;
        // std::cout << time_sum / testTime - 1 * time_standard_deviation << std::endl;
        // std::cout << time_sum / testTime - 2 * time_standard_deviation << std::endl;
        std::cout << std::endl << "RMSE" << std::endl;
        // std::cout << RMSE_sum / testTime + 2 * RMSE_standard_deviation << std::endl;
        // std::cout << RMSE_sum / testTime + 1 * RMSE_standard_deviation << std::endl;
        std::cout << RMSE_sum / testTime << std::endl;
        // std::cout << RMSE_sum / testTime - 1 * RMSE_standard_deviation << std::endl;
        // std::cout << RMSE_sum / testTime - 2 * RMSE_standard_deviation << std::endl;
        //excel
        // std::string s(argc[t]); s.resize(s.length() - 4);
        // std::ofstream oFile;
        // oFile.open("./Result/" + N_OUTLIERS_RATE_str[N_index] + "/" + s + " " + N_OUTLIERS_RATE_str[N_index] + ".csv", std::ios::out | std::ios::trunc);
        // // oFile.open(s + ".csv", std::ios::out | std::ios::trunc);
        // oFile << "Scale_Error:, (esti. / grou.)" << std::endl; 
        // oFile << "m + 2v," << scale_error_sum / testTime + 2 * scale_error_standard_deviation << std::endl;
        // oFile << "m + 1v," << scale_error_sum / testTime + 1 * scale_error_standard_deviation << std::endl;
        // oFile << "m," << scale_error_sum / testTime << std::endl;
        // oFile << "m - 1v," << scale_error_sum / testTime - 1 * scale_error_standard_deviation << std::endl;
        // oFile << "m - 2v," << scale_error_sum / testTime - 2 * scale_error_standard_deviation << std::endl;
        // oFile << std::endl << "Angle_Error:, (degree)" << std::endl;
        // oFile << "m + 2v,"<< angle_error_sum / testTime + 2 * angle_error_standard_deviation << std::endl;
        // oFile << "m + 1v,"<< angle_error_sum / testTime + 1 * angle_error_standard_deviation << std::endl;
        // oFile << "m,"<< angle_error_sum / testTime << std::endl;
        // oFile << "m - 1v,"<< angle_error_sum / testTime - 1 * angle_error_standard_deviation << std::endl;
        // oFile << "m - 2v,"<< angle_error_sum / testTime - 2 * angle_error_standard_deviation << std::endl;
        // oFile << std::endl << "Trans_Error:, (meter)" << std::endl;
        // oFile << "m + 2v,"<< trans_error_sum / testTime + 2 * trans_error_standard_deviation << std::endl;
        // oFile << "m + 1v,"<< trans_error_sum / testTime + 1 * trans_error_standard_deviation << std::endl;
        // oFile << "m,"<< trans_error_sum / testTime << std::endl;
        // oFile << "m - 1v,"<< trans_error_sum / testTime - 1 * trans_error_standard_deviation << std::endl;
        // oFile << "m - 2v,"<< trans_error_sum / testTime - 2 * trans_error_standard_deviation << std::endl;
        // oFile << std::endl << "Time:, (second)" << std::endl;
        // oFile << "m + 2v,"<< time_sum / testTime + 2 * time_standard_deviation << std::endl;
        // oFile << "m + 1v,"<< time_sum / testTime + 1 * time_standard_deviation << std::endl;
        // oFile << "m,"<< time_sum / testTime << std::endl;
        // oFile << "m - 1v,"<< time_sum / testTime - 1 * time_standard_deviation << std::endl;
        // oFile << "m - 2v,"<< time_sum / testTime - 2 * time_standard_deviation << std::endl;
        // oFile << std::endl << "RMSE" << std::endl;
        // oFile << "m + 2v,"<< RMSE_sum / testTime + 2 * RMSE_standard_deviation << std::endl;
        // oFile << "m + 1v,"<< RMSE_sum / testTime + 1 * RMSE_standard_deviation << std::endl;
        // oFile << "m,"<< RMSE_sum / testTime << std::endl;
        // oFile << "m - 1v,"<< RMSE_sum / testTime - 1 * RMSE_standard_deviation << std::endl;
        // oFile << "m - 2v,"<< RMSE_sum / testTime - 2 * RMSE_standard_deviation << std::endl;
        // oFile.close();
        // std::cout << std::string(argc[t]) << " End." << std::endl;

        sumSE += scale_best;
        sumRE += angle_best;
        sumTE += trans_best;
        sumTime += time_best;
        sumRMSE += RMSE_best;     
        // sleep(3);
      // }
    }
    // oFile.open("Average.csv", std::ios::out | std::ios::trunc);
    oFile << N_OUTLIERS_RATE_str[N_index] << std::endl;
    oFile << "Scale_Error: (esti. / grou.)" << std::endl;
    oFile << "," << sumSE / totalData << std::endl;
    oFile << "Angle_Error: (degree)" << std::endl;
    oFile << ","<< sumRE / totalData << std::endl;
    oFile << "Trans_Error: (meter)" << std::endl;
    oFile << ","<< sumTE / totalData << std::endl;
    oFile << "Time: (second)" << std::endl;
    oFile << ","<< sumTime / totalData << std::endl;
    oFile << "RMSE:" << std::endl;
    oFile << ","<< sumRMSE / totalData << std::endl << std::endl;
  }
  oFile.close();  
}
