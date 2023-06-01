#include <iostream>
#include <vector>
#include <algorithm>


// Function to dehaze hazy images
std::vector<cv::Mat> dehaze_image(std::vector<cv::Mat> hazy_images, std::vector<cv::Mat> tmaps) {
    assert(hazy_images.size() == tmaps.size());
    int num_samples = hazy_images.size();
    int num_pixels = tmaps[0].rows * tmaps[0].cols;
    int k = int(0.001 * num_pixels);
    std::vector<cv::Mat> dehazed_images;

    for (int s_idx = 0; s_idx < num_samples; s_idx++) {
        cv::Mat hazy_image = hazy_images[s_idx];
        cv::Mat tmap = tmaps[s_idx];
        cv::Mat tmap_flat = tmap.reshape(1, num_pixels);
        cv::Mat darkest_k;
        cv::sortIdx(tmap_flat, darkest_k, cv::SORT_ASCENDING | cv::SORT_EVERY_ROW);
        darkest_k = darkest_k.colRange(0, k);

        cv::Mat hi_flat = cv::mean(hazy_image.reshape(1, num_pixels), 2);
        cv::Mat airlight;
        cv::reduce(hi_flat, airlight, 1, cv::REDUCE_MAX);
        airlight = cv::repeat(airlight, 1, num_pixels);

        cv::Mat denom = cv::max(tmap, 0.1);
        cv::Mat numerator = hazy_image - airlight;
        cv::Mat dehazed = numerator / cv::repeat(denom, 1, hazy_image.channels()) + airlight;

        // Clip values between 0 and 1
        cv::Mat dehazed_clipped;
        cv::threshold(dehazed, dehazed_clipped, 0, 0, cv::THRESH_TOZERO);
        cv::threshold(dehazed_clipped, dehazed_clipped, 1, 1, cv::THRESH_TRUNC);

        dehazed_images.push_back(dehazed_clipped.reshape(hazy_image.size()));
    }

    return dehazed_images;
}
