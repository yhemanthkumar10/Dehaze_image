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

// Function to print images during training
void print_images_callback(TfLiteInterpreter* interpreter, const std::vector<float>& hazy_image, const std::vector<float>& trans_map) {
    std::cout << "Performance on random pics" << std::endl;
    std::cout << "##" << std::string(50, '#') << std::endl;
    int num_samples = hazy_image.size() / (kInputWidth * kInputHeight * 3);
    int random_pic = 10; // fixed_idx
    int num_random = 2;
    for (int j = 0; j <= num_random; j++) {
        std::vector<float> t_batch(kOutputWidth * kOutputHeight);
        std::vector<float> h_batch(kInputWidth * kInputHeight * 3);

        // Get trans_map for random_pic
        memcpy(t_batch.data(), &trans_map[random_pic * kOutputWidth * kOutputHeight], sizeof(float) * kOutputWidth * kOutputHeight);

        // Get hazy_image for random_pic
        memcpy(h_batch.data(), &hazy_image[random_pic * kInputWidth * kInputHeight * 3], sizeof(float) * kInputWidth * kInputHeight * 3);

        // Run inference
        interpreter->ResizeInputTensor(0, {1, kInputHeight, kInputWidth, 3});
        interpreter->AllocateTensors();
        float* input_data = interpreter->typed_input_tensor<float>(0);
        memcpy(input_data, h_batch.data(), sizeof(float) * kInputWidth * kInputHeight * 3);
        interpreter->Invoke();
        const float* output_data = interpreter->typed_output_tensor<float>(0);

        // Get t_pred
        std::vector<float> t_pred_npy(kOutputWidth * kOutputHeight);
        memcpy(t_pred_npy.data(), output_data, sizeof(float) * kOutputWidth * kOutputHeight);

        // Print t_pred
        std::cout << "##" << std::string(50, '#') << std::endl;

        random_pic = std::rand() % num_samples; // Generate a random pic
    }
}

int main() {
    // Define the training parameters
    int num_epochs = 50;
    int checkpoint = 10;
    bool grad_sim = true;
    int scaling_coefficient = 1000;
    int patch_size = 1;
    int batch_size = 100;
    float lr_initial = 0.01;
    float lr_decay_factor = 0.1;
    int lr_decay_interval = 100;
    float momentum = 0.9;
    float l2_weight_decay = 5e-04;
    std::string run_dir = "./saved_models/test";

    // Create the run directory if it doesn't exist
    if (!std::filesystem::exists(run_dir)) {
        std::filesystem::create_directory(run_dir);
    }

    std::string logfilename = run_dir + "/log.txt";
    std::ofstream logfile(logfilename, std::ios_base::app);
    logfile << "num_epochs=" << num_epochs << ", ";
    logfile << "checkpoint=" << checkpoint << ", ";
    logfile << "grad_sim=" << grad_sim << ", ";
    logfile << "scaling_coefficient=" << scaling_coefficient << ", ";
    logfile << "patch_size=" << patch_size << ", ";
    logfile << "batch_size=" << batch_size << ", ";
    logfile << "lr_initial=" << lr_initial << ", ";
    logfile << "lr_decay_factor=" << lr_decay_factor << ", ";
    logfile << "lr_decay_interval=" << lr_decay_interval << ", ";
    logfile << "momentum=" << momentum << ", ";
    logfile << "l2_weight_decay=" << l2_weight_decay << std::endl;
    logfile.close();

    // Load the dataset
    std::string dataset_path = "../data/nyu_hazy_trn.mat";
    // Load the dataset using h5py or any other library

    // Create the model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("mymodels.tflite");
    assert(model != nullptr);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    interpreter->AllocateTensors();

    // Get the input and output dimensions
    int kInputHeight = interpreter->input_tensor(0)->dims->data[1];
    int kInputWidth = interpreter->input_tensor(0)->dims->data[2];
    int kOutputHeight = interpreter->output_tensor(0)->dims->data[1];
    int kOutputWidth = interpreter->output_tensor(0)->dims->data[2];

    // Run the training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::cout << "Epoch #" << epoch << std::endl;

        // Check if learning rate needs to be decayed
        if (epoch % lr_decay_interval == 0) {
            if (epoch == 0) {
                // Set initial learning rate
                // Set optimizer parameters
            } else {
                // Decay learning rate
                // Set optimizer parameters
            }
        }

        // Shuffle examples

        int start_batch = 0;
        int num_samples = 100;  // Update with the actual number of samples
        std::vector<float> loss_values;
        int minibatch_num = 0;
        // Minibatch loop
        while (start_batch < num_samples) {
            int end_batch = std::min(start_batch + batch_size, num_samples);
            std::vector<float> h_batch((end_batch - start_batch) * kInputWidth * kInputHeight * 3);
            std::vector<float> t_batch((end_batch - start_batch) * kOutputWidth * kOutputHeight);

            // Get hazy_image and trans_map minibatches

            // Run inference and update model parameters

            // Compute loss and update optimizer
            float loss_batch = 0.0;  // Update with the actual loss calculation

            std::cout << "Epoch #" << epoch << ", Minibatch #" << minibatch_num << ", Loss: " << loss_batch << std::endl;

            start_batch += batch_size;
            minibatch_num++;
            loss_values.push_back(loss_batch);
        }

        // Calculate epoch time and loss mean

        // Save model checkpoint

        // Run print images callback
        print_images_callback(interpreter.get(), hazy_image, trans_map);
    }

    return 0;
}
