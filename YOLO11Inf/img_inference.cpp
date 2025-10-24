#include<iostream>
#include<opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include "YOLO11Det.hpp"
#include<fstream>
#include<iomanip>

// Cross-platform directory handling
#ifdef _WIN32
#include<direct.h>
#include<sys/stat.h>
#define MKDIR(path) _mkdir(path)
#define STAT _stat
#else
#include<sys/stat.h>
#include<sys/types.h>
#define MKDIR(path) mkdir(path, 0755)
#define STAT stat
#endif

int main() {
	try {
		std::string model_path = "C:/Users/Umidjon Shonazarov/CLionProjects/first_project/best.onnx";
		std::string image_path = "C:/Users/Umidjon Shonazarov/CLionProjects/first_project/images/ch01_2024_img1.jpg";

		std::cout << "Model path: " << model_path << std::endl;
		std::cout << "Image path: " << image_path << std::endl;

		cv::Mat img = cv::imread(image_path);
		if (img.empty()) {
			std::cerr << "Failed to load image: " << image_path << std::endl;
			return -1;
		}

		std::cout << "Image loaded: " << img.cols << "x" << img.rows << std::endl;

		YOLO11Det detector(model_path);
		detector.setConfThreshold(0.3f);
		detector.setIoUThreshold(0.45f);

		std::cout << "Model loaded successfully" << std::endl;
		std::cout << "Model input size: " << detector.getConfig().input_width << "x"
			<< detector.getConfig().input_height << std::endl;

		std::vector<Detection> detections = detector.detectImage(img);

		detector.drawDetections(img, detections);

		struct STAT info;
		if (STAT("results", &info) != 0) {
			MKDIR("results");
			std::cout << "Created 'results' folder" << std::endl;
		}

		// ===== SAVE OUTPUT IMAGE =====
		std::string output_image_path = "results/result.jpg";
		cv::imwrite(output_image_path, img);
		std::cout << "Result image saved to " << output_image_path << std::endl;

		// ===== SAVE LABELS TO CSV IN YOLO FORMAT =====
		std::string csv_path = "results/img_prop.csv";
		std::ofstream csv_file(csv_path);

		if (!csv_file.is_open()) {
			std::cerr << "Failed to create CSV file: " << csv_path << std::endl;
			return -1;
		}

		csv_file << "class_id,x_center,y_center,width,height,confidence\n";

		float img_width = static_cast<float>(img.cols);
		float img_height = static_cast<float>(img.rows);

		for (const auto& det : detections) {
			float x_center = (det.box.x + det.box.width / 2.0f) / img_width;
			float y_center = (det.box.y + det.box.height / 2.0f) / img_height;

			float norm_width = det.box.width / img_width;
			float norm_height = det.box.height / img_height;

			csv_file << det.class_id << ","
				<< std::fixed << std::setprecision(6)
				<< x_center << ","
				<< y_center << ","
				<< norm_width << ","
				<< norm_height << ","
				<< std::round(det.confidence * 100) / 100 << "\n";
		}

		csv_file.close();
		std::cout << "Labels saved to " << csv_path << " in YOLO format" << std::endl;
		std::cout << "Total detections saved: " << detections.size() << std::endl;

		cv::imshow("YOLO Result", img);
		cv::waitKey(0);

		return 0;
	}
	catch (const Ort::Exception& e) {
		std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
		return -1;
	}
	catch (const cv::Exception& e) {
		std::cerr << "OpenCV error: " << e.what() << std::endl;
		return -1;
	}
	catch (const std::exception& e) {
		std::cerr << "Standard error: " << e.what() << std::endl;
		return -1;
	}
}