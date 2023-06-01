#include <iostream>
#include <vector>
#include <random>
#include <H5Cpp.h>
#include <opencv2/opencv.hpp>
#include <Open3D/Open3D.h>

void generate_hazy_dataset(const std::string& input_file_path, const std::vector<int>& sample_idcs,
                           const std::string& output_file_path, int num_airlights, int num_scats,
                           float a_high = 1.0, float a_low = 0.7,
                           float scatt_high = 1.2, float scatt_low = 0.5,
                           const std::string& depth_dataset_name = "depths") {
    // Open input file
    H5::H5File input_file(input_file_path, H5F_ACC_RDONLY);

    // Read input dimensions
    H5::DataSet image_dataset = input_file.openDataSet("images");
    H5::DataSpace image_dataspace = image_dataset.getSpace();
    int num_samples = sample_idcs.size();
    int num_channels, width, height;
    image_dataspace.getSimpleExtentDims(&num_samples, &num_channels, &width, &height);

    // Prepare output file
    H5::H5File output_file(output_file_path, H5F_ACC_TRUNC);
    hsize_t dims[4] = {num_samples, height/2, width/2, num_channels};
    H5::DataSpace dataspace(4, dims);

    H5::DataSet airlight_dataset = output_file.createDataSet("airlight", H5::PredType::NATIVE_FLOAT, dataspace);
    H5::DataSet scatt_coeff_dataset = output_file.createDataSet("scatt_coeff", H5::PredType::NATIVE_FLOAT, dataspace);
    H5::DataSet image_dataset = output_file.createDataSet("image", H5::PredType::NATIVE_FLOAT, dataspace);
    H5::DataSet hazy_image_dataset = output_file.createDataSet("hazy_image", H5::PredType::NATIVE_FLOAT, dataspace);
    H5::DataSet trans_map_dataset = output_file.createDataSet("trans_map", H5::PredType::NATIVE_FLOAT, dataspace);

    // Generate hazy dataset
    for (int s_idx = 0; s_idx < num_samples; s_idx++) {
        int inp_idx = sample_idcs[s_idx];
        cv::Mat im;
        cv::Mat im_depth;

        // Load image data
        input_file["images"][inp_idx].read(im.data, H5::PredType::NATIVE_FLOAT);

        // Load depth data using Open3D
        open3d::geometry::Image depth_image;
        input_file[depth_dataset_name][inp_idx].read(depth_image.ptr(), H5::PredType::NATIVE_FLOAT);
        im_depth = cv::Mat(depth_image.height_, depth_image.width_, CV_32FC1, depth_image.ptr());

        // Resize depth to match hazy image dimensions
        cv::resize(im_depth, im_depth, cv::Size(width/2, height/2));

        // Set image and hazy image data pointers
        float* image_ptr = static_cast<float*>(image_data.ptr());
        float* hazy_image_ptr = static_cast<float*>(hazy_image_data.ptr());
        float* trans_map_ptr = static_cast<float*>(trans_map_data.ptr());

        // Generate hazy images
        for (int j = 0; j < num_airlights; j++) {
            float airlight = (a_high - a_low) * static_cast<float>(std::rand()) / RAND_MAX + a_low;
            airlight_dataset.write(&airlight, H5::PredType::NATIVE_FLOAT);

            for (int k = 0; k < num_scats; k++) {
                float scatt_coeff = (scatt_high - scatt_low) * static_cast<float>(std::rand()) / RAND_MAX + scatt_low;
                scatt_coeff_dataset.write(&scatt_coeff, H5::PredType::NATIVE_FLOAT);

                // Generate hazy image and transmission map
                for (int y = 0; y < height/2; y++) {
                    for (int x = 0; x < width/2; x++) {
                        float trans = 1.0f - scatt_coeff * im_depth.at<float>(y, x);
                        trans_map_ptr[y * width/2 + x] = trans;

                        for (int c = 0; c < num_channels; c++) {
                            hazy_image_ptr[(y * width/2 + x) * num_channels + c] = airlight * im_ptr[(y * width/2 + x) * num_channels + c] + (1 - airlight) * trans * im_ptr[(y * width/2 + x) * num_channels + c];
                        }
                    }
                }

                // Write image, hazy image, and transmission map to output file
                image_dataset.write(image_data, H5::PredType::NATIVE_FLOAT);
                hazy_image_dataset.write(hazy_image_data, H5::PredType::NATIVE_FLOAT);
                trans_map_dataset.write(trans_map_data, H5::PredType::NATIVE_FLOAT);
            }
        }
    }

    // Close files
    input_file.close();
    output_file.close();
}

int main() {
    std::string input_file_path = "path/to/input/file.h5";
    std::string output_file_path = "path/to/output/file.h5";
    std::vector<int> sample_idcs = {0, 1, 2, 3, 4};  // Example sample indices

    int num_airlights = 1;
    int num_scats = 3;
    float a_high = 1.0;
    float a_low = 0.7;
    float scatt_high = 1.2;
    float scatt_low = 0.5;

    std::string depth_dataset_name = "depths";

    generate_hazy_dataset(input_file_path, sample_idcs, output_file_path, num_airlights, num_scats,
                          a_high, a_low, scatt_high, scatt_low, depth_dataset_name);

    return 0;
}
