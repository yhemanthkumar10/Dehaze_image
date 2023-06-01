#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>
#include <cassert>
#include <cstring>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/optional_debug_tools.h>

// Function to compare models on images
void compare_models_on_images(std::map<std::string, std::unique_ptr<tflite::Interpreter>>& models,
                              const std::vector<float>& hazy_images,
                              const std::vector<float>& true_tmaps) {
    int num_rows = idcs.size();
    int num_cols = 1 + models.size();
    int figh = 12;
    int figw = 15;
    // Create figure and axes
    // Note: You need to have a library to handle plotting, such as OpenCV or another graphics library
    //       Here, we assume you have a library to handle plotting and have the necessary functions available.
    create_figure(num_rows, num_cols, figh, figw);

    std::vector<std::string> mnames;
    for (const auto& model : models) {
        mnames.push_back(model.first);
    }

    for (int i = 0; i < idcs.size(); i++) {
        std::vector<float> orig_img(hazy_images.begin() + (idcs[i] * kInputWidth * kInputHeight * 3),
                                    hazy_images.begin() + ((idcs[i] + 1) * kInputWidth * kInputHeight * 3));
        std::vector<float> hazy_img_batch(hazy_images.begin() + (idcs[i] * kInputWidth * kInputHeight * 3),
                                          hazy_images.begin() + ((idcs[i] + 1) * kInputWidth * kInputHeight * 3));

        // Plot hazy image
        plot_image(hazy_img_batch, i, 0);

        for (int j = 0; j < mnames.size(); j++) {
            if (mnames[j] == "ground_truth") {
                std::vector<float> tmap_batch(true_tmaps.begin() + (idcs[i] * kOutputWidth * kOutputHeight),
                                              true_tmaps.begin() + ((idcs[i] + 1) * kOutputWidth * kOutputHeight));
                // Plot tmap_batch if required
            } else {
                std::unique_ptr<tflite::Interpreter>& model = models[mnames[j]];

                // Run inference
                std::vector<float> pred_tmaps(kOutputWidth * kOutputHeight);
                memcpy(pred_tmaps.data(), input_data, sizeof(float) * kOutputWidth * kOutputHeight);

                // Plot pred_tmaps if required

                std::vector<float> dehazed(kInputWidth * kInputHeight * 3);
                // Call dehaze_image function and store result in dehazed

                // Plot dehazed image
                plot_image(dehazed, i, j + 1);
            }
        }
    }

    // Set labels and titles

    // Save the figure or display it
}

// Function to compute PSNR for a batch of images
float compute_psnr_batch(const std::vector<float>& images1, const std::vector<float>& images2) {
    std::vector<float> values;
    for (int i = 0; i < images1.size(); i += kInputWidth * kInputHeight * 3) {
        float psnr_val = psnr(images1.data() + i, images2.data() + i);
        values.push_back(psnr_val);
    }
    float mean_psnr = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    return mean_psnr;
}

// Function to compute SSIM for a batch of images
float compute_ssim_batch(const std::vector<float>& images1, const std::vector<float>& images2,
                         bool multichannel) {
    std::vector<float> values;
    for (int i = 0; i < images1.size(); i += kInputWidth * kInputHeight * 3) {
        float ssim_val = ssim(images1.data() + i, images2.data() + i, multichannel);
        values.push_back(ssim_val);
    }
    float mean_ssim = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    return mean_ssim;
}

// Function to compute metrics for models
std::vector<std::string> compute_metrics(std::map<std::string, std::unique_ptr<tflite::Interpreter>>& models,
                                         const std::vector<float>& hazy_images,
                                         const std::vector<float>& true_tmaps) {
    std::vector<std::string> mnames;
    std::vector<float> original_images;
    std::vector<float> best_dehazed;
    for (const auto& model : models) {
        mnames.push_back(model.first);
    }

    // Compute metrics for each model
    for (int i = 0; i < mnames.size(); i++) {
        std::unique_ptr<tflite::Interpreter>& model = models[mnames[i]];

        // Run inference for hazy_images and store the results in pred_tmaps

        // Call dehaze_image function and store result in dehazed_images

        // Compute metrics using the computed values and store in metrics
        float psnr_orig = compute_psnr_batch(original_images, dehazed_images);
        float ssim_orig = compute_ssim_batch(original_images, dehazed_images, true);
        float psnr_best = compute_psnr_batch(best_dehazed, dehazed_images);
        float ssim_best = compute_ssim_batch(best_dehazed, dehazed_images, true);
        float psnr_tmap = compute_psnr_batch(pred_tmaps, true_tmaps);
        float ssim_tmap = compute_ssim_batch(pred_tmaps, true_tmaps, false);

        metrics.push_back({mnames[i], psnr_orig, ssim_orig, psnr_best, ssim_best, psnr_tmap, ssim_tmap});
    }

    return metrics;
}

int main() {
    std::string trn_path = "../data/nyu_hazy_trn.mat";
    std::string tst_path = "../data/nyu_hazy_tst.mat";
    std::string path_to_hazy = "../data/nyu_hazy.mat";

    // Load trn_file and tst_file using h5py or any other library

    std::map<std::string, std::unique_ptr<tflite::Interpreter>> models;

    std::map<std::string, std::string> model_paths = {
            {"base", "../saved_models/base/model_50"},
            {"patch_1", "../saved_models/p1_g0/model_50"},
            {"patch_3", "../saved_models/p3_g0/model_50"},
            {"patch_5", "../saved_models/p5_g0/model_50"}
    };

    for (const auto& entry : model_paths) {
        const std::string& key = entry.first;
        const std::string& mpath = entry.second;
        // Load model using TensorFlow Lite
        std::unique_ptr<tflite::Interpreter> model = LoadModel(mpath);
        models[key] = std::move(model);
    }

    std::vector<int> random_idcs = getRandomIndices(tst_file["hazy_image"].shape[0], 5);

    compare_models_on_images(models, tst_file, random_idcs, true);
    compare_models_on_images(models, tst_file, random_idcs, false);

    return 0;
}
