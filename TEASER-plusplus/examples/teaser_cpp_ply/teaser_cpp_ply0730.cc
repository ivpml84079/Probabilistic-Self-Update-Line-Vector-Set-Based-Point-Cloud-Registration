#include <iostream>
#include <iomanip>
#include <filesystem>
#include <string>
#include <numeric>
#include <memory>
#include <fstream>
#include <random>
#include <vector>
#include <chrono> // 20250730
#include <cmath> // 20250730
// pcl
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/pcl_search.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/conversions.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/io.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
// eigen3
#include <Eigen/Dense>

#include <teaser/ply_io.h>
#include <teaser/registration.h>
#include <teaser/matcher.h>


// Macro constants for generating noise and outliers
#define NOISE_BOUND 0.025

int loadPLYFile(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>& cloud_out) {
    pcl::PCLPointCloud2 cloud_blob;
    pcl::PLYReader reader;

    // 1. 讀檔
    int ret = reader.read(filename, cloud_blob);
    if (ret < 0) return ret;

    // 2. 直接從 fields 提取欄位名稱
    bool has_x = false, has_y = false, has_z = false;
    for (const auto& field : cloud_blob.fields) {
        if (field.name == "x") has_x = true;
        if (field.name == "y") has_y = true;
        if (field.name == "z") has_z = true;
    }

    if (!(has_x && has_y && has_z)) {
        std::cerr << "Missing x/y/z fields in PLY file.\n";
        return -2;
    }

    // 3. 嘗試轉型為 PointXYZ（會自動做 float 轉換）
    try {
        pcl::PointCloud<pcl::PointXYZ> temp_cloud;
        pcl::fromPCLPointCloud2(cloud_blob, temp_cloud);
        cloud_out.swap(temp_cloud);
    } catch (...) {
        std::cerr << "Failed to convert PCLPointCloud2 to PointXYZ.\n";
        return -3;
    }

    return 0;
}

// VoxelGridDownsample
int sampleLeafsized(
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& down_cloud,
	float downsample_size
) {
	pcl::PointCloud<pcl::PointXYZ> cloud_sub;
	pcl::PointCloud<pcl::PointXYZ> cloud_out;
	float leafsize = downsample_size * (std::pow(static_cast <int64_t> (std::numeric_limits <int32_t>::max()) - 1, 1. / 3.) - 1);

	pcl::octree::OctreePointCloud<pcl::PointXYZ> oct(leafsize); // new octree structure
	oct.setInputCloud(cloud_in);
	oct.defineBoundingBox();
	oct.addPointsFromInputCloud();

	pcl::VoxelGrid<pcl::PointXYZ> vg; // new voxel grid filter
	vg.setLeafSize(downsample_size, downsample_size, downsample_size);
	vg.setInputCloud(cloud_in);

	size_t num_leaf = oct.getLeafCount();

	pcl::octree::OctreePointCloud<pcl::PointXYZ>::LeafNodeIterator it = oct.leaf_depth_begin(), it_e = oct.leaf_depth_end();
	for (size_t i = 0; i < num_leaf; ++i, ++it)
	{
		pcl::IndicesPtr ids(new std::vector <int>); // extract octree leaf points
		pcl::octree::OctreePointCloud<pcl::PointXYZ>::LeafNode* node = (pcl::octree::OctreePointCloud <pcl::PointXYZ>::LeafNode*) * it;
		node->getContainerPtr()->getPointIndices(*ids);

		vg.setIndices(ids); // set cloud indices
		vg.filter(cloud_sub); // filter cloud

		cloud_out += cloud_sub; // add filter result
	}

	down_cloud->clear();
	*down_cloud += cloud_out;
	return (static_cast <int> (cloud_out.size())); // return number of points in sampled cloud
}

void issKeyPointExtration(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr ISS,
    pcl::PointIndicesPtr ISS_Idx, double resolution
) {
	double iss_salient_radius_ = 6 * resolution;
	double iss_non_max_radius_ = 4 * resolution;
	//double iss_non_max_radius_ = 2 * resolution;//for office
	//double iss_non_max_radius_ = 9 * resolution;//for railway
	double iss_gamma_21_(0.975);
	double iss_gamma_32_(0.975);
	double iss_min_neighbors_(5);
	int iss_threads_(1); //switch to the number of threads in your cpu for acceleration

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;

	iss_detector.setSearchMethod(tree);
	iss_detector.setSalientRadius(iss_salient_radius_);
	iss_detector.setNonMaxRadius(iss_non_max_radius_);
	iss_detector.setThreshold21(iss_gamma_21_);
	iss_detector.setThreshold32(iss_gamma_32_);
	iss_detector.setMinNeighbors(iss_min_neighbors_);
	iss_detector.setNumberOfThreads(iss_threads_);
	iss_detector.setInputCloud(cloud);
	iss_detector.compute(*ISS);
	ISS_Idx->indices = iss_detector.getKeypointsIndices()->indices;
	ISS_Idx->header = iss_detector.getKeypointsIndices()->header;
}

void fpfhComputation(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double resolution, 
    pcl::PointIndicesPtr iss_Idx, pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_out
) {
	//compute normal
	pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>());
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(3 * resolution);
	ne.compute(*normal);

	//compute fpfh using normals
	pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
	fpfh_est.setInputCloud(cloud);
	fpfh_est.setInputNormals(normal);
	fpfh_est.setSearchMethod(tree);
	fpfh_est.setRadiusSearch(8 * resolution);
	fpfh_est.setNumberOfThreads(1);
	fpfh_est.setIndices(iss_Idx);
	fpfh_est.compute(*fpfh_out);
}

int loadPointCloudPair(
    const std::string &srcPointCloudPath,
    const std::string &tgtPointCloudPath,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt
) {
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(srcPointCloudPath, *cloud_src) < 0 ||
        pcl::io::loadPLYFile<pcl::PointXYZ>(tgtPointCloudPath, *cloud_tgt) < 0) {
        return -1;
    }
    return 0;
}
// 使用 TEASER++ 內建的 FPFH Matcher 來建立對應點
void build_correspondences_simple(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_src,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_tgt,
                                  Eigen::Matrix<double, 3, Eigen::Dynamic>& src_corr_pts,
                                  Eigen::Matrix<double, 3, Eigen::Dynamic>& dst_corr_pts) {

  teaser::FPFHMatcher::Params params;
  // 我們可以使用預設參數，或者微調
  // params.normal_radius_multiplier = ...;
  // params.feature_radius_multiplier = ...;

  teaser::FPFHMatcher matcher(params);
  matcher.setInputClouds(cloud_src, cloud_tgt);
  auto correspondences = matcher.getCorrespondences();

  // 填充 Eigen 矩陣
  src_corr_pts.resize(3, correspondences->size());
  dst_corr_pts.resize(3, correspondences->size());

  for (size_t i = 0; i < correspondences->size(); ++i) {
    int src_idx = correspondences->at(i).index_query;
    int tgt_idx = correspondences->at(i).index_match;
    src_corr_pts.col(i) << cloud_src->points[src_idx].x, cloud_src->points[src_idx].y,
        cloud_src->points[src_idx].z;
    dst_corr_pts.col(i) << cloud_tgt->points[tgt_idx].x, cloud_tgt->points[tgt_idx].y,
        cloud_tgt->points[tgt_idx].z;
  }
}
/*
void correspondence_builder(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt,
    float noise_bound,
    const std::string& output_dir,
    const std::string& pair_name  // 新增参数：点云对名称

    // --- 新增這兩行 ---20250730
    Eigen::Matrix<double, 3, Eigen::Dynamic>& src_corrs_out,
    Eigen::Matrix<double, 3, Eigen::Dynamic>& tgt_corrs_out
    // --- 新增結束 ---

) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr down_cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr down_cloud_tgt(new pcl::PointCloud<pcl::PointXYZ>);

    // 使用Voxel Grid filter进行下採樣
    sampleLeafsized(cloud_src, down_cloud_src, noise_bound);
    sampleLeafsized(cloud_tgt, down_cloud_tgt, noise_bound);

    // Compute ISS
    pcl::PointCloud<pcl::PointXYZ>::Ptr issS(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndicesPtr issIdxS(new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZ>::Ptr issT(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndicesPtr issIdxT(new pcl::PointIndices);
    std::cout << "extracting ISS keypoints... " << noise_bound << std::endl;
    issKeyPointExtration(down_cloud_src, issS, issIdxS, noise_bound);
    issKeyPointExtration(down_cloud_tgt, issT, issIdxT, noise_bound);
    issS->is_dense = false;
    issT->is_dense = false;
    std::cout << "size of issS = " << issS->size() << std::endl;
    std::cout << "size of issT = " << issT->size() << std::endl;
    
    // Compute FPFH
    std::cout << "computing fpfh..." << std::endl;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhS(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhT(new pcl::PointCloud<pcl::FPFHSignature33>());
    fpfhComputation(down_cloud_src, noise_bound, issIdxS, fpfhS);
    fpfhComputation(down_cloud_tgt, noise_bound, issIdxT, fpfhT);

    std::cout << "match correspondences..." << std::endl;
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
    pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> est;
    est.setInputSource(fpfhS);
    est.setInputTarget(fpfhT);
    est.determineCorrespondences(*correspondences);
    int corrNum = correspondences->size();
    std::cout << "Found " << corrNum << " correspondences." << std::endl;

    // 创建输出目录（如果不存在）
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    // 输出correspondence到文件
    std::string corr_file = output_dir + "/" + pair_name + ".txt";
    std::ofstream output_file(corr_file);
    
    if (!output_file.is_open()) {
        return;
    }
    
    // 写入头部信息：对应点总数
    output_file << corrNum << std::endl;
    //20250730 sw 註解掉下面兩行
    //Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, corrNum);
    //Eigen::Matrix<double, 3, Eigen::Dynamic> tgt(3, corrNum);
    src_corrs_out.resize(3, corrNum); // 新增這行：調整大小 20250730 sw
    tgt_corrs_out.resize(3, corrNum); // 新增這行：調整大小 20250730 sw


    for (size_t i = 0; i < corrNum; ++i) {
        int si = correspondences->at(i).index_query;
        int ti = correspondences->at(i).index_match;
        src_corrs_out.col(i) << issS->at(si).x, issS->at(si).y, issS->at(si).z;
        tgt_corrs_out.col(i) << issT->at(ti).x, issT->at(ti).y, issT->at(ti).z;
        
        // 输出格式：source点坐标 target点坐标
        output_file << std::fixed << std::setprecision(6)
                    << src.col(i)(0) << " " << src.col(i)(1) << " " << src.col(i)(2) << " "
                    << tgt.col(i)(0) << " " << tgt.col(i)(1) << " " << tgt.col(i)(2) << std::endl;
    }
    
    output_file.close();
}
*/
// --- 新增：讀取 Ground Truth (GT) 轉換的函數 ---
// 這裡假設你的 GT 檔案格式為：R11 R12 R13 T1 R21 R22 R23 T2 R31 R32 R33 T3 (12個double值)
// 你需要根據你實際的 GT 文件格式來修改這個函數
bool loadGroundTruthTransform(const std::string& gt_file_path, Eigen::Matrix3d& R_gt,
                              Eigen::Vector3d& T_gt) {
  std::ifstream gt_stream(gt_file_path);
  if (!gt_stream.is_open()) {
    std::cerr << "Error: Could not open ground truth file: " << gt_file_path << std::endl;
    return false;
  }

  // 假设GT文件是 4x4 矩阵，或者直接是 R 和 T 的值
  // 这里简单地假设读取 12 个 double 值，前9个是 R，后3个是 T
  // 如果你的 GT 文件格式不同，请修改这里
  double val;
  std::vector<double> vals;
  while (gt_stream >> val) {
    vals.push_back(val);
  }
  gt_stream.close();

  if (vals.size() < 12) { // 至少需要 3x3 R 和 3x1 T
    std::cerr << "Error: Not enough values in ground truth file: " << gt_file_path << std::endl;
    return false;
  }

  R_gt << vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8];
  T_gt << vals[9], vals[10], vals[11];

  return true;
}





int main() {
    std::cout << "Starting point cloud correspondence generation" << std::endl;

    // 基础路径
    std::string dataset_path = "./poor_sj/dataset/office";
    std::string pairs_file = dataset_path + "/pairs.txt";
    std::string gt_dir = dataset_path + "/gt_transforms";
    
    // 创建输出目录
    std::string output_dir = "./output";
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
        std::cout << "Created output directory: " << output_dir << std::endl;
    }
    
    // 检查pairs.txt文件是否存在
    if (!std::filesystem::exists(pairs_file)) {
        std::cerr << "Pairs file not found: " << pairs_file << std::endl;
        return -1;
    }
    
    // 读取pairs.txt文件
    std::ifstream pairs_stream(pairs_file);
    if (!pairs_stream.is_open()) {
        std::cerr << "Failed to open pairs file: " << pairs_file << std::endl;
        return -1;
    }
    
    std::cout << "Reading pairs from: " << pairs_file << std::endl;

    // --- 定义要测试的 RANSAC 迭代次數 ---
    std::vector<size_t> ransac_iterations_to_test = {3, 4, 5, 6,
                                                     7, 8, 9, 10}; // 擴大範圍，更清楚趨勢

    // --- 打開一個 CSV 文件來記錄所有結果 ---
    std::ofstream results_csv("experiment_results.csv");
    if (!results_csv.is_open()) {
      std::cerr << "Error: Could not open experiment_results.csv for writing." << std::endl;
      return -1;
    }
    // 寫入 CSV 標題行
    results_csv << "Pair_Name,RANSAC_Iterations,Rot_Error_Rad,Trans_Error_M,Time_s,Num_Outliers,"
                   "Final_Inlier_Count"
                << std::endl;
    



    
    std::string src_name, tgt_name;
    while (pairs_stream >> src_name >> tgt_name) {
        // 提取数字部分（移除's'并转换为整数）
        int src_id = std::stoi(src_name.substr(1));
        int tgt_id = std::stoi(tgt_name.substr(1));
        
        // 构建完整的文件路径
        std::string src_file = dataset_path + "/" + src_name + ".ply";
        std::string tgt_file = dataset_path + "/" + tgt_name + ".ply";


        // 构建输出文件名格式 (src_id-tgt_id) 20250730
        std::string pair_name = std::to_string(src_id) + "-" + std::to_string(tgt_id);

        std::cout << "\n--- Processing pair: " << pair_name << " ---\n";
        //20250730
        
        // 检查文件是否存在
        if (!std::filesystem::exists(src_file)) {
            std::cerr << "Source file not found: " << src_file << std::endl;
            continue;
        }
        if (!std::filesystem::exists(tgt_file)) {
            std::cerr << "Target file not found: " << tgt_file << std::endl;
            continue;
        }
        /*
        // 构建输出文件名格式 (src_id-tgt_id)
        std::string pair_name = std::to_string(src_id) + "-" + std::to_string(tgt_id);
        
        std::cout << "Processing pair: " << pair_name << " (source: " << src_name 
                  << ".ply, target: " << tgt_name << ".ply)" << std::endl;
        */
        // 加载点云对
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt(new pcl::PointCloud<pcl::PointXYZ>);
        
        if (loadPointCloudPair(src_file, tgt_file, cloud_src, cloud_tgt) != 0) {
            std::cerr << "Failed to load point cloud pair: " << pair_name << std::endl;
            continue;
        }
        /*
        // 生成 correspondence，并输出到指定目录
        std::cout << "Building correspondences for pair: " << pair_name << std::endl;
        correspondence_builder(cloud_src, cloud_tgt, NOISE_BOUND, output_dir, pair_name);
        
        std::cout << "Completed processing pair: " << pair_name << std::endl;
    }
    
    pairs_stream.close();
    std::cout << "All point cloud correspondences generated successfully" << std::endl;
    return 0;
    */
        // --- 生成 Correspondence (特徵點對應) ---
        // 這些矩陣將填充實際的對應點座標
        Eigen::Matrix<double, 3, Eigen::Dynamic> src_corr_pts;
        Eigen::Matrix<double, 3, Eigen::Dynamic> dst_corr_pts;
        
        /*
        std::cout << "Building correspondences for pair: " << pair_name << std::endl;
        correspondence_builder(cloud_src, cloud_tgt, NOISE_BOUND, output_dir, pair_name,
                               src_corr_pts, dst_corr_pts);
        std::cout << "Found " << src_corr_pts.cols() << " correspondences for pair: " << pair_name
                  << std::endl;
        */
        std::cout << "Building correspondences using TEASER's matcher for pair: " << pair_name
                  << std::endl;
        build_correspondences_simple(cloud_src, cloud_tgt, src_corr_pts, dst_corr_pts);



        // --- 載入 Ground Truth 變換 ---
        Eigen::Matrix3d expected_R;
        Eigen::Vector3d expected_t;
        std::string gt_file_current_pair =
            gt_dir + "/" + pair_name + ".txt"; // 假設 GT 檔案名與 pair_name 相同
        if (!loadGroundTruthTransform(gt_file_current_pair, expected_R, expected_t)) {
          std::cerr << "Error: Failed to load ground truth for pair: " << pair_name
                    << ". Skipping this pair." << std::endl;
          continue;
        }

        // --- 針對不同的 RANSAC 迭代次數進行實驗 ---
        for (size_t current_ransac_iter : ransac_iterations_to_test) {
          std::cout << "  Running with RANSAC Iterations = " << current_ransac_iter << std::endl;

          // 初始化 TEASER++ 求解器參數
          teaser::RobustRegistrationSolver::Params params;
          params.noise_bound = NOISE_BOUND;
          params.cbar2 = 1.0;
          params.estimate_scaling = true;       // 假設你需要估計尺度
          params.rotation_max_iterations = 100; // 這是 GNC 內部的迭代次數，不是 RANSAC
          params.rotation_gnc_factor = 1.4;
          params.rotation_estimation_algorithm =
              teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
          params.rotation_cost_threshold = 0.005;

          // --- 設定我們修改過的 RANSAC 迭代次數參數 ---
          params.ransac_max_iterations = current_ransac_iter;

          // 初始化 TEASER++ 求解器
          // 注意：每次循環都要重新初始化求解器，以確保參數被正確設置
          teaser::RobustRegistrationSolver solver(params);

          // 執行配準並測量時間
          auto start_time = std::chrono::steady_clock::now();
          teaser::RegistrationSolution solution = solver.solve(src_corr_pts, dst_corr_pts);
          auto end_time = std::chrono::steady_clock::now();

          double time_taken_s =
              std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time)
                  .count();

          // 計算誤差
          double rotation_error_rad = std::acos(std::fmin(
              std::fmax(((expected_R.transpose() * solution.rotation).trace() - 1) / 2, -1.0),
              1.0));
          double translation_error_m = (expected_t - solution.translation).norm();

          // 獲取內點數（如果 solution.final_inlier_count 是最終內點數）
          int num_outliers_found = src_corr_pts.cols() - solution.final_inlier_count;

          // 將結果寫入 CSV 文件
          results_csv << pair_name << "," << current_ransac_iter << "," << rotation_error_rad << ","
                      << translation_error_m << "," << time_taken_s << "," << num_outliers_found
                      << ","                                       // 记录找到的离群点数量
                      << solution.final_inlier_count << std::endl; // 记录最终内点数量

          std::cout << "    Result: Rot Error (rad)=" << rotation_error_rad
                    << ", Trans Error (m)=" << translation_error_m << ", Time (s)=" << time_taken_s
                    << ", Inliers=" << solution.final_inlier_count
                    << ", Outliers=" << num_outliers_found << std::endl;
        }
    }

    pairs_stream.close();
    results_csv.close(); // 關閉 CSV 文件
    std::cout << "All experiments completed and results saved to experiment_results.csv"
              << std::endl;
    return 0;
}