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
#include <sstream>
#include <unistd.h> 
#include <cmath>
#include <stdlib.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

// Macro constants for generating noise and outliers
#define NOISE_BOUND 0.05
#define PI 3.1415926

int ddtime = 10;
int unknownScale = 0;
std::string test_data = "3DM";
// std::string test_data = "3DLM";
// std::string test_data = "KITTI";
// std::string descriptor = "fpfh";
std::string descriptor = "fcgf";

std::string data_source = "./dataset/real/";

std::string threeDMatch[8] = {
  "sun3d-hotel_umd-maryland_hotel1",
  "sun3d-hotel_uc-scan3",
  "sun3d-mit_76_studyroom-76-1studyroom2",
  "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
	"7-scenes-redkitchen",
	"sun3d-home_at-home_at_scan1_2013_jan_1",
	"sun3d-home_md-home_md_scan9_2012_sep_30",
	"sun3d-hotel_umd-maryland_hotel3",
};

std::string threeDlomatch[8] = {
	"7-scenes-redkitchen_3dlomatch",
	"sun3d-home_at-home_at_scan1_2013_jan_1_3dlomatch",
	"sun3d-home_md-home_md_scan9_2012_sep_30_3dlomatch",
	"sun3d-hotel_uc-scan3_3dlomatch",
	"sun3d-hotel_umd-maryland_hotel1_3dlomatch",
	"sun3d-hotel_umd-maryland_hotel3_3dlomatch",
	"sun3d-mit_76_studyroom-76-1studyroom2_3dlomatch",
	"sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_3dlomatch",
};


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
  int stdDevMultiplier = 1;

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

  double o_height_mean = std::accumulate(o_histogram_height.begin(), o_histogram_height.end(), 0.0) / o_histogram_height.size();
  double o_height_variance = 0.0;
  for (int height : o_histogram_height) {
    o_height_variance += std::pow(height - o_height_mean, 2);
  }
  double o_height_stdDev = std::sqrt(o_height_variance / o_histogram_height.size());
  double o_height_threshold = o_height_mean + stdDevMultiplier * o_height_stdDev;

  for(int i = 0; i < angles_histogram.size(); i++) {
    if(std::abs(i - peak_id) > 2){
      for(int j = 0; j < angles_histogram[i].size(); j++) {
        keep_mask[angles_histogram[i][j]] = -1;
      }
    }
    if(angles_histogram[i].size() > o_height_threshold) {
      for(int j = 0; j < angles_histogram[i].size(); j++) {
        keep_mask[angles_histogram[i][j]] = 1;
        remain_count++;
      }
    }
  }

  std::cout << "remain_count: " << remain_count << std::endl;
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

int main(int argv, char* argc[]) {
  if(test_data == "3DM")
  {
    std::string data_path = data_source + "3dmatch_3dlomatch/";
    std::string result_path = "./Result/3DMATCH/";
    std::ofstream avgResultFile;
    avgResultFile.open(result_path + "Average_" + descriptor + "_" + std::to_string(unknownScale) + ".csv", std::ios::out | std::ios::trunc);
    avgResultFile << "ScaleError,AngleError,TransError,RMSE,Time,SuccessRate" << std::endl;
    
    std::vector<int> happened_total_time, local_count_time, basic_count_time;
    std::map<int, int> happened;
    for(int i = 0; i < 8; i++)
    {
      avgResultFile << threeDMatch[i];
      std::string file_path = data_path + threeDMatch[i] + "/";
      std::string pair_label_file = file_path + "gt.log";
      std::ifstream labelfile(pair_label_file);
      std::vector<std::pair<int, int>> labels;
      std::string line;
      while (std::getline(labelfile, line))
      {
          std::istringstream iss(line);
          int label1, label2, value;
          if (iss >> label1 >> label2 >> value) {
              labels.push_back(std::make_pair(label1, label2));
          }
      }
      labelfile.close();
      avgResultFile << ",Total Pair," << labels.size() << std::endl;
      
      //used for draw
      Eigen::Matrix4d bestTransform = Eigen::Matrix4d::Identity();
      std::string bestPair = "";
      double bestRMSE = 1000000000000000, bestAngleError = 1000000000000000, bestTransError = 100000000000000, bestTimeError = 10000000000, bestScaleError = 10000000000;

      std::ofstream resultFile;
      resultFile.open(result_path + threeDMatch[i] + "_" + descriptor + "_" + std::to_string(unknownScale) + ".csv", std::ios::out | std::ios::trunc);
      resultFile << "ScaleError,AngleError,TransError,RMSE,Time,SuccessRate" << std::endl;
      double scaleErrorSum = 0, angleErrorSum = 0, transErrorSum = 0, RMSESum = 0, timeErrorSum = 0;
      int successCount = 0;      

      for(auto label : labels)
      {
        resultFile << label.first << "," << label.second << std::endl;
        std::cout << label.first << "\t" << label.second << std::endl;
        std::string corr_file = file_path + "cloud_bin_" + std::to_string(label.first) + "+cloud_bin_" + std::to_string(label.second);
        if(descriptor == "fpfh")
          corr_file += "@corr.txt";
        else if(descriptor == "fcgf")
          corr_file += "@corr_fcgf.txt";

        std::ifstream corrFile(corr_file);
        Eigen::Matrix<double, 3, Eigen::Dynamic> src, tgt;
        while (std::getline(corrFile, line))
        {
            std::istringstream iss(line);
            double src1, src2, src3, tgt1, tgt2, tgt3;
            if (iss >> src1 >> src2 >> src3 >> tgt1 >> tgt2 >> tgt3)
            {
              src.conservativeResize(3, src.cols() + 1);
              tgt.conservativeResize(3, tgt.cols() + 1);
              src.col(src.cols() - 1) << src1, src2, src3;
              tgt.col(tgt.cols() - 1) << tgt1, tgt2, tgt3;
            }
        }
        corrFile.close();
        
        std::string gt_file = file_path + "cloud_bin_" + std::to_string(label.first) + "+cloud_bin_" + std::to_string(label.second);
        if(descriptor == "fpfh")
          gt_file += "@GTmat.txt";
        else if(descriptor == "fcgf")
          gt_file += "@GTmat_fcgf.txt";
        Eigen::Matrix4d T;
        std::ifstream gtFile(gt_file);
        for (int l = 0; l < 4; ++l)
        {
          double a, b, c, d;
          if (gtFile >> a >> b >> c >> d)
              T.row(l) << a, b, c, d;
        }
        gtFile.close();
      
        srand( time(NULL) );
        //random
        double testScale = 1;
        if (unknownScale)
        {
          testScale = 1 + 4 * ((1.0 - 0.0) * rand() / (RAND_MAX + 1.0) + 0.0);
          tgt = testScale * tgt;
        }
        
        Eigen::Matrix4d onceTransform = Eigen::Matrix4d::Identity();
        double onceRMSE = 1000000000000000, onceAngleError = 1000000000000000, onceTransError = 100000000000000, onceTimeError = 10000000000, onceScaleError = 10000000000;
        for(int t = 0; t < ddtime; t++)
        {
          std::cout << "Test: " << t << std::endl;
          //start solving
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
          //end solving

          double scaleError = std::fabs(testScale - solution.scale);
          double angleError = getAngularError(T.topLeftCorner(3, 3), solution.rotation);
          double transError = (T.topRightCorner(3, 1) - solution.translation).norm();
          double timeError = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
          Eigen::Matrix4d TRANSFORM;
          TRANSFORM <<  1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1;
          TRANSFORM.topLeftCorner(3, 3) = solution.rotation;
          TRANSFORM.topRightCorner(3, 1) = solution.translation;
          // Homogeneous coordinates
          Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
          src_h.resize(4, src.cols());
          src_h.topRows(3) = src;
          src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(src.cols());
          Eigen::Matrix<double, 4, Eigen::Dynamic> src_gt_h = testScale * T * src_h;
          Eigen::Matrix<double, 3, Eigen::Dynamic> src_gt = src_gt_h.topRows(3);
          // Apply transformation
          Eigen::Matrix<double, 3, Eigen::Dynamic> src_solve = (solution.scale * TRANSFORM * src_h).topRows(3);
          double RMSE = 0.0;
          for(int m = 0; m < src_solve.cols(); m++)
          {
            RMSE += pow((src_gt.col(m) - src_solve.col(m)).norm(), 2);
          }
          RMSE = sqrt(RMSE / src_solve.cols());

          if(unknownScale)
          {
            std::cout << "Expected scale: " << std::endl;
            std::cout << testScale << std::endl;
            std::cout << "Estimated scale: " << std::endl;
            std::cout << solution.scale << std::endl;
            std::cout << "Error: " << scaleError  << std::endl << std::endl;
          }
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
          std::cout << std::endl;
          std::cout << "Error (m): " << transError << std::endl;
          std::cout << "RMSE: " << std::endl;
          std::cout << RMSE << std::endl;
          std::cout << std::endl;
          std::cout << "Time taken (s): " << timeError << std::endl;

          if(RMSE < onceRMSE)
          {
            onceRMSE = RMSE;
            onceTransform = TRANSFORM;
            onceAngleError = angleError;
            onceTransError = transError;
            onceTimeError = timeError;
            onceScaleError = scaleError;
          }  
        }

        resultFile << onceScaleError << "," << onceAngleError << "," << onceTransError << "," << onceRMSE << "," << onceTimeError << std::endl;
        if(onceScaleError <= 0.1 && onceAngleError <= 15 && onceTransError <= 0.3 && onceTimeError <= 60.0)
        {
          scaleErrorSum += onceScaleError;
          angleErrorSum += onceAngleError;
          transErrorSum += onceTransError;
          RMSESum += onceRMSE;
          timeErrorSum += onceTimeError;
          successCount++;
        }

        //used for draw
        if(onceRMSE < bestRMSE)
        {
          bestTransform = onceTransform;
          bestRMSE = onceRMSE;
          bestAngleError = onceAngleError;
          bestTransError = onceTransError;
          bestTimeError = onceTimeError;
          bestScaleError = onceScaleError;
          bestPair = std::to_string(label.first) + "_" + std::to_string(label.second);
        }
        
        std::ifstream ifs("static.txt");
        std::string line2;
        int times[3];
        for (int j = 0; j < 3; j++)
        {
          std::getline(ifs, line2);
          std::istringstream iss(line2);
          iss >> times[j];
        }
        happened_total_time.push_back(times[0]);
        local_count_time.push_back(times[1]);
        basic_count_time.push_back(times[2]);
        std::getline(ifs, line2);
        std::istringstream iss2(line2);
        int happened_basic;
        while(iss2 >> happened_basic)
        {
          happened[happened_basic]++;
        }
        ifs.close();
      }
      resultFile.close();
      avgResultFile << scaleErrorSum / successCount << "," << angleErrorSum / successCount << "," << transErrorSum / successCount << "," 
                  << RMSESum / successCount << "," << timeErrorSum / successCount << "," << successCount * 1.0 / labels.size() << std::endl;

      //used for draw
      std::ofstream outfile;
      outfile.open(result_path + "bestTransform_" + threeDMatch[i] + "_" + descriptor + "_" + std::to_string(unknownScale) + ".csv", std::ios::out | std::ios::trunc);
      // 將變數寫入文件
      outfile << "Best Pair: " << bestPair << std::endl;
      outfile << "Best Transform:\n" << bestTransform << std::endl;
      outfile << "Best Scale Error: " << bestScaleError << std::endl;
      outfile << "Best Angle Error: " << bestAngleError << std::endl;
      outfile << "Best Translation Error: " << bestTransError << std::endl;
      outfile << "Best RMSE: " << bestRMSE << std::endl;
      outfile << "Best Time Error: " << bestTimeError << std::endl;
      // 關閉文件
      outfile.close();
    }
    avgResultFile.close();

    int happenedSum = 0, localSum = 0, basicSum = 0;
    for(int i = 0; i < happened_total_time.size(); i++)
    {
      happenedSum += happened_total_time[i];
      localSum += local_count_time[i];
      basicSum += basic_count_time[i];
    }
    double averageSimilarIter = 0;
    for(int i = 1; i <= 10; i++)
    {
      averageSimilarIter += i * happened[i];
    }
    averageSimilarIter /= happenedSum;
    double host_local_similar = happened[1], local_local_similar = happenedSum - happened[1];    
    std::cout << "writing summary.txt\n";
    std::ofstream ofs;
    ofs.open("summary.txt", std::ios::out);
    ofs << "總共有幾個Local: " << localSum << std::endl;
    ofs << "總共有幾個basic: " << basicSum << std::endl;
    ofs << "總共發生幾次答案相近: " << happenedSum << std::endl;
    ofs << "總共發生幾次Host與Local答案相近: " << host_local_similar << std::endl;
    ofs << "總共發生幾次Local與Local答案相近: " << local_local_similar << std::endl;
    ofs << "Host與Local答案相近佔總Local個數的比例: " << host_local_similar / localSum << std::endl;
    ofs << "Local與Local答案相近佔總Local個數的比例: " << local_local_similar / localSum << std::endl;
    ofs << "Host與Local答案相近佔全部答案相近次數的比例: " << host_local_similar / happenedSum << std::endl;
    ofs << "Local與Local答案相近佔全部答案相近次數的比例: " << local_local_similar / happenedSum << std::endl;
    ofs << "答案相近發生的iteration平均: " << averageSimilarIter << std::endl;
    ofs.close();
    std::cout << "end writing summary.txt\n";
  }
  else if(test_data == "KITTI")
  {
    std::string data_path = data_source + "KITTI/";
    std::string result_path = "./Result/KITTI";
    if(descriptor == "fpfh")
    {
      data_path += "correspondence_fpfh/";
      result_path += "_FPFH/";
    }
    else if(descriptor == "fcgf")
    {
      data_path += "correspondence_fcgf/";
      result_path += "_FCGF/";
    }

    //used for draw
    Eigen::Matrix4d bestTransform = Eigen::Matrix4d::Identity();
    std::string bestPair = "";
    double bestRMSE = 1000000000000000, bestAngleError = 1000000000000000, bestTransError = 100000000000000, bestTimeError = 10000000000, bestScaleError = 10000000000;
    Eigen::Matrix4d bestTransform2 = Eigen::Matrix4d::Identity();
    std::string bestPair2 = "";
    double bestRMSE2 = 1000000000000000, bestAngleError2 = 1000000000000000, bestTransError2 = 100000000000000, bestTimeError2 = 10000000000, bestScaleError2 = 10000000000;

    double scaleErrorSum = 0, angleErrorSum = 0, transErrorSum = 0, RMSESum = 0, timeErrorSum = 0;
    int successCount = 0;
    std::vector<double> scaleErrors(555, 0), angleErrors(555, 0), transErrors(555, 0), RMSEs(555, 0), timeErrors(555, 0);
    std::vector<int> success(555, 0);
    for(int i = 0; i < 555; i++)
    {
      std::cout << i << std::endl;
      std::string file_path = data_path + std::to_string(i) + "/";
      std::string corr_file = file_path;
      std::string gt_file = file_path;
      if(descriptor == "fpfh")
      {
        corr_file += "fpfh@corr.txt";
        gt_file += "fpfh@gtmat.txt";
      }
      else if(descriptor == "fcgf")
      {
        corr_file += "fcgf@corr.txt";
        gt_file += "fcgf@gtmat.txt";
      }
      //read correspondence file
      std::ifstream corrFile(corr_file);
      Eigen::Matrix<double, 3, Eigen::Dynamic> src, tgt;
      std::string line;
      while (std::getline(corrFile, line))
      {
          std::istringstream iss(line);
          double src1, src2, src3, tgt1, tgt2, tgt3;
          if (iss >> src1 >> src2 >> src3 >> tgt1 >> tgt2 >> tgt3)
          {
            src.conservativeResize(3, src.cols() + 1);
            tgt.conservativeResize(3, tgt.cols() + 1);
            src.col(src.cols() - 1) << src1, src2, src3;
            tgt.col(tgt.cols() - 1) << tgt1, tgt2, tgt3;
          }
      }
      corrFile.close();
      // std::cout << src.cols() << " " << tgt.cols() << std::endl;
      //read ground truth file
      Eigen::Matrix4d T;
      std::ifstream gtFile(gt_file);
      for (int l = 0; l < 4; ++l)
      {
        double a, b, c, d;
        if (gtFile >> a >> b >> c >> d)
            T.row(l) << a, b, c, d;
      }
      gtFile.close();
      
      srand( time(NULL) );
      //random
      double testScale = 1;
      if (unknownScale)
      {
        testScale = 1 + 4 * ((1.0 - 0.0) * rand() / (RAND_MAX + 1.0) + 0.0);
        tgt = testScale * tgt;
      }

      Eigen::Matrix4d onceTransform = Eigen::Matrix4d::Identity();
      double onceRMSE = 1000000000000000, onceAngleError = 1000000000000000, onceTransError = 100000000000000, onceTimeError = 10000000000, onceScaleError = 10000000000;
      for(int t = 0; t < ddtime; t++)
      {
        std::cout << "Test: " << t << std::endl;
        //start solving
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
        //end solving

        double scaleError = std::fabs(testScale - solution.scale);
        double angleError = getAngularError(T.topLeftCorner(3, 3), solution.rotation);
        double transError = (T.topRightCorner(3, 1) - solution.translation).norm();
        double timeError = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
        Eigen::Matrix4d TRANSFORM;
        TRANSFORM <<  1, 0, 0, 0,
                      0, 1, 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1;
        TRANSFORM.topLeftCorner(3, 3) = solution.rotation;
        TRANSFORM.topRightCorner(3, 1) = solution.translation;
        // Homogeneous coordinates
        Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
        src_h.resize(4, src.cols());
        src_h.topRows(3) = src;
        src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(src.cols());
        Eigen::Matrix<double, 4, Eigen::Dynamic> src_gt_h = testScale * T * src_h;
        Eigen::Matrix<double, 3, Eigen::Dynamic> src_gt = src_gt_h.topRows(3);
        // Apply transformation
        Eigen::Matrix<double, 3, Eigen::Dynamic> src_solve = (solution.scale * TRANSFORM * src_h).topRows(3);
        double RMSE = 0.0;
        for(int m = 0; m < src_solve.cols(); m++)
        {
          RMSE += pow((src_gt.col(m) - src_solve.col(m)).norm(), 2);
        }
        RMSE = sqrt(RMSE / src_solve.cols());

        if(unknownScale)
        {
          std::cout << "Expected scale: " << std::endl;
          std::cout << testScale << std::endl;
          std::cout << "Estimated scale: " << std::endl;
          std::cout << solution.scale << std::endl;
          std::cout << "Error: " << scaleError  << std::endl << std::endl;
        }
        std::cout << "Expected rotation: " << std::endl;
        std::cout << T.topLeftCorner(3, 3) << std::endl;
        std::cout << "Estimated rotation: " << std::endl;
        std::cout << solution.rotation << std::endl;
        std::cout << "Error (deg): " << angleError  << std::endl;
        std::cout << std::endl;
        std::cout << "Expected translation: " << std::endl;
        std::cout << T.topRightCorner(3, 1) << std::endl;
        std::cout << "Estimated translation: " << std::endl;
        std::cout << solution.translation << std::endl;
        std::cout << std::endl;
        std::cout << "Error (m): " << transError << std::endl;
        std::cout << "RMSE: " << std::endl;
        std::cout << RMSE << std::endl;
        std::cout << std::endl;
        std::cout << "Time taken (s): " << timeError << std::endl;

        if(RMSE < onceRMSE)
        {
          onceRMSE = RMSE;
          onceTransform = TRANSFORM;
          onceAngleError = angleError;
          onceTransError = transError;
          onceTimeError = timeError;
          onceScaleError = scaleError;
        }  
      }

      scaleErrors[i] = onceScaleError;
      angleErrors[i] = onceAngleError;
      transErrors[i] = onceTransError;
      RMSEs[i] = onceRMSE;
      timeErrors[i] = onceTimeError;
      if(onceScaleError <= 0.1 && onceAngleError <= 5 && onceTransError <= 0.6 && onceTimeError <= 60.0)
      {
        scaleErrorSum += onceScaleError;
        angleErrorSum += onceAngleError;
        transErrorSum += onceTransError;
        RMSESum += onceRMSE;
        timeErrorSum += onceTimeError;
        successCount++;
        success[i] = 1;
      }

      //used for draw
      if(onceRMSE < bestRMSE)
      {
        bestTransform2 = bestTransform;
        bestRMSE2 = bestRMSE;
        bestAngleError2 = bestAngleError;
        bestTransError2 = bestTransError;
        bestTimeError2 = bestTimeError;
        bestScaleError2 = bestScaleError;
        bestPair2 = bestPair;
        bestTransform = onceTransform;
        bestRMSE = onceRMSE;
        bestAngleError = onceAngleError;
        bestTransError = onceTransError;
        bestTimeError = onceTimeError;
        bestScaleError = onceScaleError;
        bestPair = std::to_string(i);
      }
      else if(onceRMSE < bestRMSE2)
      {
        bestTransform2 = onceTransform;
        bestRMSE2 = onceRMSE;
        bestAngleError2 = onceAngleError;
        bestTransError2 = onceTransError;
        bestTimeError2 = onceTimeError;
        bestScaleError2 = onceScaleError;
        bestPair2 = std::to_string(i);
      }      

      // if(i == 0)
      // {        
      //   i = 553;
      //   continue;
      // }
    }

    //used for draw
    std::ofstream outfile;
    outfile.open(result_path + "bestTransform_" + descriptor + "_" + std::to_string(unknownScale) + ".csv", std::ios::out | std::ios::trunc);
    outfile << "Best Pair: " << bestPair << std::endl;
    outfile << "Best Transform:\n" << bestTransform << std::endl;
    outfile << "Best Scale Error: " << bestScaleError << std::endl;
    outfile << "Best Angle Error: " << bestAngleError << std::endl;
    outfile << "Best Translation Error: " << bestTransError << std::endl;
    outfile << "Best RMSE: " << bestRMSE << std::endl;
    outfile << "Best Time Error: " << bestTimeError << std::endl;
    outfile.close();
    outfile.open(result_path + "bestTransform2_" + descriptor + "_" + std::to_string(unknownScale) + ".csv", std::ios::out | std::ios::trunc);
    outfile << "Best Pair: " << bestPair2 << std::endl;
    outfile << "Best Transform:\n" << bestTransform2 << std::endl;
    outfile << "Best Scale Error: " << bestScaleError2 << std::endl;
    outfile << "Best Angle Error: " << bestAngleError2 << std::endl;
    outfile << "Best Translation Error: " << bestTransError2 << std::endl;
    outfile << "Best RMSE: " << bestRMSE2 << std::endl;
    outfile << "Best Time Error: " << bestTimeError2 << std::endl;
    outfile.close();

    //write result file
    std::ofstream resultFile;
    resultFile.open(result_path + "KITTI_" + descriptor + "_" + std::to_string(unknownScale) + ".csv", std::ios::out | std::ios::trunc);
    resultFile << "Error(scale),Error(deg),Error(m),RMSE,time(s),SuccessRate" << std::endl;
    resultFile << "Avg." << std::endl;
    resultFile << scaleErrorSum / 555.0 << "," << angleErrorSum / 555.0 << "," << transErrorSum / 555.0 << "," << RMSESum / 555.0 
              << "," << timeErrorSum / 555.0 << "," << successCount / 555.0 << std::endl;
    for(int i = 0; i < 555; i++)
    {
      resultFile << std::to_string(i) << std::endl;
      resultFile << scaleErrors[i] << "," << angleErrors[i] << "," << transErrors[i] << "," << RMSEs[i] << "," << timeErrors[i] << "," << success[i] << std::endl;
    }
    resultFile.close();
  }
}