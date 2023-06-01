#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/version.h>
#include <iostream>

class SurfaceSmoothnessLoss {
public:
    SurfaceSmoothnessLoss(int patch_size = 1) : patch_size(patch_size) {
        // Initialize filters
        _init_filters();
    }

    void _init_filters() {
        diff_x = {
                1, 0, -1,
                2, 0, -2,
                1, 0, -1
        };
        diff_y = diff_x;
    }

    void forward(float* z, int width, int height, float* loss) {
        assert(z != nullptr);
        assert(loss != nullptr);

        std::vector<float> z_x(width * height, 0.0);
        std::vector<float> z_y(width * height, 0.0);
        std::vector<float> n_z(width * height, 0.0);
        std::vector<float> n_x(width * height, 0.0);
        std::vector<float> n_y(width * height, 0.0);
        std::vector<float> normal(width * height * 3, 0.0);
        *loss = 0.0;

        // Calculate z_x and z_y
        _conv2d(z, width, height, diff_x, 3, 3, z_x.data());
        _conv2d(z, width, height, diff_y, 3, 3, z_y.data());

        // Calculate n_z, n_x, n_y
        for (int i = 0; i < width * height; i++) {
            n_z[i] = 1 / std::sqrt(1 + z_x[i] * z_x[i] + z_y[i] * z_y[i]);
            n_x[i] = n_z[i] * z_x[i];
            n_y[i] = n_z[i] * z_y[i];
        }

        // Concatenate n_x, n_y, n_z to form normal
        for (int i = 0; i < width * height; i++) {
            normal[i] = n_x[i];
            normal[width * height + i] = n_y[i];
            normal[2 * width * height + i] = n_z[i];
        }

        // Calculate surface smoothness loss
        for (int i = 1; i <= patch_size; i++) {
            for (int j = 1; j <= patch_size; j++) {
                float loss_sum = 0.0;
                for (int k = 0; k < width * height; k++) {
                    int idx1 = (i + (k / width)) * width + j + (k % width);
                    int idx2 = (k / width) * width + (k % width);
                    float dot_prod = 0.0;
                    for (int c = 0; c < 3; c++) {
                        dot_prod += normal[idx1 + c * width * height] * normal[idx2 + c * width * height];
                    }
                    float cos_dist = 1.0 - dot_prod;
                    loss_sum += cos_dist;
                }
                *loss += loss_sum / (width * height);
            }
        }
    }

private:
    void _conv2d(const float* input, int width, int height, const std::vector<float>& kernel, int kernel_width, int kernel_height, float* output) {
        // Convolution operation
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float sum = 0.0;
                for (int m = 0; m < kernel_height; m++) {
                    for (int n = 0; n < kernel_width; n++) {
                        int input_row = i - kernel_height / 2 + m;
                        int input_col = j - kernel_width / 2 + n;
                        if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                            sum += input[input_row * width + input_col] * kernel[m * kernel_width + n];
                        }
                    }
                }
                output[i * width + j] = sum;
            }
        }
    }

    std::vector<float> diff_x;
    std::vector<float> diff_y;
    int patch_size;
};

class SumLoss {
public:
    SumLoss(double scaling_coefficient = 100, int patch_size = 1) : scaling_coefficient(scaling_coefficient),
                                                                    patch_size(patch_size) {
    }

    void forward(float* z, float* z_true, int width, int height, float* loss) {
        assert(z != nullptr);
        assert(z_true != nullptr);
        assert(loss != nullptr);

        // Calculate main loss
        float mse_loss = 0.0;
        for (int i = 0; i < width * height; i++) {
            mse_loss += (z[i] - z_true[i]) * (z[i] - z_true[i]);
        }
        mse_loss /= width * height;

        // Calculate auxiliary loss
        SurfaceSmoothnessLoss aux_loss(patch_size);
        float surface_loss = 0.0;
        aux_loss.forward(z, width, height, &surface_loss);

        // Calculate total loss
        *loss = mse_loss + scaling_coefficient * surface_loss;
    }

private:
    double scaling_coefficient;
    int patch_size;
};

class GradientSimilarityOptimizer {
public:
    GradientSimilarityOptimizer(double learning_rate, double momentum, double weight_decay,
                                double scaling_coefficient = 100, int patch_size = 1)
            : learning_rate(learning_rate),
              momentum(momentum),
              weight_decay(weight_decay),
              aligned(true),
              scaling_coefficient(scaling_coefficient),
              patch_size(patch_size),
              mse(),
              aux_loss(patch_size) {
        reset_optimizer();
    }

    void reset_optimizer() {
        // Reset the optimizer
        optimizer = tf::lite::FloatSGDOptimizer(learning_rate, momentum, weight_decay);
    }

    bool optimize_loss(float* z, float* z_true, int width, int height, float* main_loss, float* auxiliary_loss) {
        assert(z != nullptr);
        assert(z_true != nullptr);
        assert(main_loss != nullptr);
        assert(auxiliary_loss != nullptr);

        // Calculate main and auxiliary losses
        sum_loss.forward(z, z_true, width, height, main_loss);
        aux_loss.forward(z, width, height, auxiliary_loss);

        std::vector<std::vector<float>> grads(2, std::vector<float>());
        for (int i = 0; i < 2; i++) {
            grads[i].resize(width * height, 0.0);
            optimizer.ResetGradsAndVars();
            if (i == 1)
                aux_loss.forward(z, width, height, nullptr);  // RetainGraph
            else
                sum_loss.forward(z, z_true, width, height, nullptr);
            optimizer.CalculateGradients();

            for (const auto& grad : optimizer.GetGradients()) {
                grads[i].push_back(grad);
            }
        }

        double dot_prod = 0.0;
        std::vector<int> include = {0};
        for (int j = 0; j < grads[0].size(); j++) {
            dot_prod += grads[1][j] * grads[0][j];
        }

        if (dot_prod > 0) {
            std::cout << "aux_loss is in alignment" << std::endl;
            aligned = true;
            include.push_back(1);
        } else {
            std::cout << "aux_loss is NOT in alignment" << std::endl;
            if (aligned)
                reset_optimizer();
            aligned = false;
        }

        optimizer.ResetGradsAndVars();
        for (int i : include) {
            for (int j = 0; j < grads[i].size(); j++) {
                optimizer.AddGradAndVar(grads[i][j], &z[j]);
            }
        }

        optimizer.ApplyGradients();

        return aligned;
    }

private:
    double learning_rate;
    double momentum;
    double weight_decay;
    bool aligned;
    double scaling_coefficient;
    int patch_size;
    tf::lite::FloatSGDOptimizer optimizer;
    SumLoss sum_loss;
    SurfaceSmoothnessLoss aux_loss;
};

int main() {
    // Load TensorFlow Lite model
    std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile("model.tflite");

    // Create TensorFlow Lite interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);

    // Allocate tensors
    interpreter->AllocateTensors();

    // Get input and output tensors
    TfLiteTensor* input_tensor = interpreter->tensor(interpreter->inputs()[0]);
    TfLiteTensor* output_tensor = interpreter->tensor(interpreter->outputs()[0]);

    // Get input dimensions
    int width = input_tensor->dims->data[1];
    int height = input_tensor->dims->data[2];

    // Initialize inputs and outputs
    float* input_data = reinterpret_cast<float*>(input_tensor->data.raw);
    float* output_data = reinterpret_cast<float*>(output_tensor->data.raw);
    float* z_true = new float[width * height];  // Initialize with ground truth depth values

    // Run inference
    interpreter->Invoke();

    // Process the output
    // ...

    delete[] z_true;

    return 0;
}
