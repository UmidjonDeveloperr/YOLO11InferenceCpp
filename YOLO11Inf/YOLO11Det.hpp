#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <memory>

struct YOLO11Config {
    float conf_threshold = 0.3f;
    float iou_threshold = 0.45f;
    int line_thickness = 2;
    int input_width = 640;
    int input_height = 640;
};

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

class YOLO11Det {
public:
    YOLO11Det() : config_(), env_(nullptr), session_(nullptr) {
    }

    YOLO11Det(const YOLO11Config& config) : config_(config), env_(nullptr), session_(nullptr) {
    }

    YOLO11Det(const std::string& model_path, const YOLO11Config& config = YOLO11Config())
        : config_(config), env_(nullptr), session_(nullptr) {
        initializeSession(model_path);
    }

#ifdef _WIN32
    YOLO11Det(const std::wstring& model_path, const YOLO11Config& config = YOLO11Config())
        : config_(config), env_(nullptr), session_(nullptr) {
        initializeSession(model_path);
    }
#endif

    ~YOLO11Det() {}

    // IMAGE PREPROCESSING
    std::vector<float> preprocessImage(const cv::Mat& img, int input_width, int input_height) {
        cv::Mat resized, rgb;

        cv::resize(img, resized, cv::Size(input_width, input_height));

        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

        std::vector<float> input_tensor_values(3 * input_height * input_width);
        std::vector<cv::Mat> channels(3);
        cv::split(rgb, channels);

        size_t channel_size = input_height * input_width;
        for (size_t c = 0; c < 3; ++c) {
            std::memcpy(input_tensor_values.data() + c * channel_size,
                channels[c].data,
                channel_size * sizeof(float));
        }

        return input_tensor_values;
    }

    /// Preprocess image using configured input size
    std::vector<float> preprocessImage(const cv::Mat& img) {
        return preprocessImage(img, config_.input_width, config_.input_height);
    }

    // POST-PROCESSING
    std::vector<int> applyNMS(const std::vector<cv::Rect>& boxes,
        const std::vector<float>& confidences,
        float iou_threshold) {
        std::vector<int> indices;
        std::vector<std::pair<float, int>> sorted_confidences;

        for (size_t i = 0; i < confidences.size(); ++i) {
            sorted_confidences.push_back(std::make_pair(confidences[i], static_cast<int>(i)));
        }
        std::sort(sorted_confidences.begin(), sorted_confidences.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });

        std::vector<bool> suppressed(confidences.size(), false);

        for (size_t i = 0; i < sorted_confidences.size(); ++i) {
            int idx = sorted_confidences[i].second;

            if (suppressed[idx]) continue;

            indices.push_back(idx);

            for (size_t j = i + 1; j < sorted_confidences.size(); ++j) {
                int idx2 = sorted_confidences[j].second;

                if (suppressed[idx2]) continue;

                float iou = calculateIoU(boxes[idx], boxes[idx2]);
                if (iou > iou_threshold) {
                    suppressed[idx2] = true;
                }
            }
        }

        return indices;
    }

    // apply NMS using configured IoU threshold
    std::vector<int> applyNMS(const std::vector<cv::Rect>& boxes,
        const std::vector<float>& confidences) {
        return applyNMS(boxes, confidences, config_.iou_threshold);
    }

    void drawDetection(cv::Mat& img,
        const Detection& detection,
        const std::string& label = "",
        const cv::Scalar& color = cv::Scalar(0, 255, 0)) {
        cv::rectangle(img, detection.box, color, config_.line_thickness);

        std::string display_label = label;
        if (display_label.empty()) {
            std::ostringstream label_stream;
            label_stream << "Class " << detection.class_id << ": "
                << std::fixed << std::setprecision(2) << detection.confidence;
            display_label = label_stream.str();
        }

        drawLabel(img, display_label,
            cv::Point(detection.box.x, detection.box.y + detection.box.height),
            color);
    }

    void drawDetections(cv::Mat& img,
        const std::vector<Detection>& detections,
        const std::vector<std::string>& class_names = {}) {
        for (const auto& detection : detections) {
            cv::Scalar color = cv::Scalar(0, 255, 0);

            std::string label;
            if (!class_names.empty() && detection.class_id < static_cast<int>(class_names.size())) {
                std::ostringstream label_stream;
                label_stream << class_names[detection.class_id] << ": "
                    << std::fixed << std::setprecision(2) << detection.confidence;
                label = label_stream.str();
            }

            drawDetection(img, detection, label, color);
        }
    }

    YOLO11Config getConfig() const {
        return config_;
    }

    void setConfig(const YOLO11Config& config) {
        config_ = config;
    }

    void setConfThreshold(float threshold) {
        config_.conf_threshold = threshold;
    }

    void setIoUThreshold(float threshold) {
        config_.iou_threshold = threshold;
    }

    void setInputSize(int width, int height) {
        config_.input_width = width;
        config_.input_height = height;
    }

    std::vector<std::string> getInputNames() const {
        if (!session_) {
            return {};
        }

        std::vector<std::string> names;
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session_->GetInputCount();

        for (size_t i = 0; i < num_inputs; ++i) {
            auto name_ptr = session_->GetInputNameAllocated(i, allocator);
            names.push_back(name_ptr.get());
        }

        return names;
    }

    std::vector<std::string> getOutputNames() const {
        if (!session_) {
            return {};
        }

        std::vector<std::string> names;
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_outputs = session_->GetOutputCount();

        for (size_t i = 0; i < num_outputs; ++i) {
            auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
            names.push_back(name_ptr.get());
        }

        return names;
    }

    std::vector<Detection> detect(float* output_data,
        const std::vector<int64_t>& output_shape,
        int img_width,
        int img_height) {
        float scale_x = static_cast<float>(img_width) / config_.input_width;
        float scale_y = static_cast<float>(img_height) / config_.input_height;

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        int num_channels = static_cast<int>(output_shape[1]);
        int num_detections = static_cast<int>(output_shape[2]);
        int num_classes = num_channels - 4;

        for (int i = 0; i < num_detections; ++i) {
            float max_class_score = 0.0f;
            int best_class_id = 0;

            for (int c = 0; c < num_classes; ++c) {
                int class_channel_idx = 4 + c;
                float class_score = output_data[class_channel_idx * num_detections + i];
                if (class_score > max_class_score) {
                    max_class_score = class_score;
                    best_class_id = c;
                }
            }

            if (max_class_score < config_.conf_threshold) continue;

            float cx = output_data[0 * num_detections + i];
            float cy = output_data[1 * num_detections + i];
            float w = output_data[2 * num_detections + i];
            float h = output_data[3 * num_detections + i];

            float x1 = std::max(0.0f, std::min((cx - w / 2.0f) * scale_x, static_cast<float>(img_width - 1)));
            float y1 = std::max(0.0f, std::min((cy - h / 2.0f) * scale_y, static_cast<float>(img_height - 1)));
            float x2 = std::max(0.0f, std::min((cx + w / 2.0f) * scale_x, static_cast<float>(img_width - 1)));
            float y2 = std::max(0.0f, std::min((cy + h / 2.0f) * scale_y, static_cast<float>(img_height - 1)));

            int box_width = static_cast<int>(x2 - x1);
            int box_height = static_cast<int>(y2 - y1);

            if (box_width > 0 && box_height > 0) {
                boxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1), box_width, box_height));
                confidences.push_back(max_class_score);
                class_ids.push_back(best_class_id);
            }
        }

        std::vector<int> indices = applyNMS(boxes, confidences);

        std::vector<Detection> detections;
        detections.reserve(indices.size());

        for (int idx : indices) {
            Detection det;
            det.box = boxes[idx];
            det.confidence = confidences[idx];
            det.class_id = class_ids[idx];
            detections.push_back(det);
        }

        return detections;
    }

    std::vector<Detection> detectImage(const cv::Mat& img) {
        if (!session_) {
            throw std::runtime_error("ONNX session not initialized. Use constructor with model path.");
        }

        std::vector<float> input_tensor_values = preprocessImage(img);

        std::vector<int64_t> input_tensor_shape = { 1, 3, config_.input_height, config_.input_width };
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_tensor_shape.data(), input_tensor_shape.size());

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
        auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);

        const char* input_names[] = { input_name_ptr.get() };
        const char* output_names[] = { output_name_ptr.get() };

        auto output_tensors = session_->Run(Ort::RunOptions{ nullptr },
            input_names, &input_tensor, 1, output_names, 1);

        float* output_data = output_tensors.front().GetTensorMutableData<float>();
        auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

        return detect(output_data, output_shape, img.cols, img.rows);
    }

private:

    YOLO11Config config_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;

    void initializeSession(const std::string& model_path) {
        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLO11Det");

            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
            std::wstring model_path_w(model_path.begin(), model_path.end());
            session_ = std::make_unique<Ort::Session>(*env_, model_path_w.c_str(), session_options);
#else
            session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
#endif

            Ort::AllocatorWithDefaultOptions allocator;
            auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            config_.input_width = static_cast<int>(input_shape[3]);
            config_.input_height = static_cast<int>(input_shape[2]);
        }
        catch (const Ort::Exception& e) {
            throw std::runtime_error(std::string("Failed to initialize ONNX Runtime session: ") + e.what());
        }
    }

#ifdef _WIN32
    void initializeSession(const std::wstring& model_path) {
        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLO11Det");

            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

            Ort::AllocatorWithDefaultOptions allocator;
            auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            config_.input_width = static_cast<int>(input_shape[3]);
            config_.input_height = static_cast<int>(input_shape[2]);
        }
        catch (const Ort::Exception& e) {
            throw std::runtime_error(std::string("Failed to initialize ONNX Runtime session: ") + e.what());
        }
    }
#endif

    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) const {
        int x1 = std::max(box1.x, box2.x);
        int y1 = std::max(box1.y, box2.y);
        int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
        int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

        int intersection_width = std::max(0, x2 - x1);
        int intersection_height = std::max(0, y2 - y1);
        int intersection_area = intersection_width * intersection_height;

        int box1_area = box1.width * box1.height;
        int box2_area = box2.width * box2.height;
        int union_area = box1_area + box2_area - intersection_area;

        return union_area > 0 ? static_cast<float>(intersection_area) / union_area : 0.0f;
    }

    void drawLabel(cv::Mat& img,
        const std::string& label,
        const cv::Point& position,
        const cv::Scalar& color) {
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        int label_y = position.y + label_size.height + 5;

        cv::rectangle(img,
            position,
            cv::Point(position.x + label_size.width, label_y),
            color, -1);

        cv::putText(img, label,
            cv::Point(position.x, label_y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(255, 255, 255), 1.5);
    }
};