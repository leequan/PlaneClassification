#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <math.h>
#include <algorithm>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "../include/matrix.h"
#include "../externals/EasyBMP/include/EasyBMP_BMP.h"
#include "../externals/EasyBMP/include/EasyBMP_DataStructures.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP *, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string &data_file, TFileList *file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;

    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0, char_idx + 1);

    while (!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList &file_list, TDataSet *data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
        // Create image
        BMP *image = new BMP();
        // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
        // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList &file_list,
                     const TLabels &labels,
                     const string &prediction_file) {
    // Check that list of files and list of labels has equal size
    assert(file_list.size() == labels.size());
    // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

    // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

Matrix<float> make_grayscale(BMP *image) {
    uint cols = image->TellHeight(), rows = image->TellWidth();
    Matrix<float> res_image(rows, cols);
    for (uint i = 0; i < rows; i++)
        for (uint j = 0; j < cols; j++) {
            RGBApixel pixel = image->GetPixel(i, j);
            float s = 0.299 * pixel.Red + 0.587 * pixel.Blue
                      + 0.114 * pixel.Green;
            res_image(i, j) = s;
        }
    return res_image;
}

Matrix<float> vertical_convolution(const Matrix<float> &gray_matrix) {
    uint rows = gray_matrix.n_rows, cols = gray_matrix.n_cols;
    Matrix<float> edge_matrix(rows + 2, cols);
    for (uint j = 0; j < cols; j++) {
        edge_matrix(0, j) = gray_matrix(0, j);
        edge_matrix(rows + 1, j) = gray_matrix(rows - 1, j);
    }
    for (uint i = 0; i < rows; i++)
        for (uint j = 0; j < cols; j++) {
            edge_matrix(i + 1, j) = gray_matrix(i, j);
        }
    Matrix<float> conv_matrix(rows, cols);
    for (uint i = 1; i < rows + 1; i++)
        for (uint j = 0; j < cols; j++) {
            int s = edge_matrix(i - 1, j) - edge_matrix(i + 1, j);
            conv_matrix(i - 1, j) = s;
        }
    return conv_matrix;
}

Matrix<float> horizontal_convolution(const Matrix<float> &gray_matrix) {
    uint rows = gray_matrix.n_rows, cols = gray_matrix.n_cols;
    Matrix<float> edge_matrix(rows, cols + 2);
    for (uint i = 0; i < rows; i++) {
        edge_matrix(i, 0) = gray_matrix(i, 0);
        edge_matrix(i, cols + 1) = gray_matrix(i, cols - 1);
    }
    for (uint i = 0; i < rows; i++)
        for (uint j = 0; j < cols; j++) {
            edge_matrix(i, j + 1) = gray_matrix(i, j);
        }
    Matrix<float> conv_matrix(rows, cols);
    for (uint i = 0; i < rows; i++)
        for (uint j = 1; j < cols + 1; j++) {
            int s = -edge_matrix(i, j - 1) + edge_matrix(i, j + 1);
            conv_matrix(i, j - 1) = s;
        }
    return conv_matrix;
}

Matrix<float> module(const Matrix<float> &hor_matrix, const Matrix<float> &vert_matrix) {
    uint rows = hor_matrix.n_rows, cols = hor_matrix.n_cols;
    Matrix<float> module_matrix(rows, cols);
    for (uint i = 0; i < rows; i++)
        for (uint j = 0; j < cols; j++) {
            int x = hor_matrix(i, j), y = vert_matrix(i, j);
            module_matrix(i, j) = sqrt(x * x + y * y);
        }
    return module_matrix;
}

Matrix<float> angle(const Matrix<float> &hor_matrix, const Matrix<float> &vert_matrix) {
    uint rows = hor_matrix.n_rows, cols = hor_matrix.n_cols;
    Matrix<float> angle_matrix(rows, cols);
    for (uint i = 0; i < rows; i++)
        for (uint j = 0; j < cols; j++) {
            int x = hor_matrix(i, j), y = vert_matrix(i, j);
            angle_matrix(i, j) = atan2(y, x);
        }
    return angle_matrix;
}

vector<float> cell_histogram(const Matrix<float> module, const Matrix<float> angle,
                             uint r1, uint r2, uint c1, uint c2) {
    // делим гистограммы на num_segments сегментов
    uint num_segments = 32;
    vector<float> hist(num_segments, 0);
    uint k;
    for (uint i = r1; i < r2; i++)
        for (uint j = c1; j < c2; j++) {
            double interval = (2 * M_PI) / num_segments;
            for (k = 0; k < num_segments; k++) {
                if (angle(i, j) >= -M_PI + k * interval &&
                    angle(i, j) <= -M_PI + (k + 1) * interval) {
                    break;
                }
            }
            // k имеет номер сегмента
            hist[k] += module(i, j);
        }
    return hist;
}


vector<float> get_all_histograms(const Matrix<float> module, const Matrix<float> angle) {
    uint rows = module.n_rows, cols = module.n_cols;
    // делим изображение клетки, их кол-во: num_row х num_col
    vector<float> res;
    uint r1, r2, c1, c2;
    uint num_row = 4, num_col = 4;
    uint row_interval, col_interval;
    if (rows % num_row == 0)
        row_interval = round(rows / num_row);
    else
        row_interval = round(rows / num_row) + 1;
    if (cols % num_col == 0)
        col_interval = round(cols / num_col);
    else
        col_interval = round(cols / num_col) + 1;
    for (uint i = 0; i < num_row; i++)
        for (uint j = 0; j < num_col; j++) {
            r1 = i * row_interval;
            r2 = (i + 1) * row_interval;
            if (r2 >= rows) {
                r2 = rows;
            }
            c1 = j * col_interval;
            c2 = (j + 1) * col_interval;
            if (c2 >= cols) {
                c2 = cols;
            }
            auto hist = cell_histogram(module, angle, r1, r2, c1, c2);
            res.insert(res.end(), hist.begin(), hist.end());
        }
    return res;
}

void norm_histogram(vector<float> &hist) {
    // используется L2-норма
    float norm = 0;
    for (uint i = 0; i < hist.size(); i++) {
        norm += hist[i] * hist[i];
    }
    norm = sqrt(norm);
    for (uint i = 0; i < hist.size(); i++) {
        hist[i] /= norm;
    }
}

vector<float> cell_color(BMP *image, uint r1, uint r2, uint c1, uint c2) {
    vector<float> res(3, 0);
    float red = 0, blue = 0, green = 0;
    uint norm = 1;
    for (uint i = r1; i < r2; i++)
        for (uint j = c1; j < c2; j++) {
            RGBApixel pixel = image->GetPixel(i, j);
            red += pixel.Red;
            blue += pixel.Blue;
            green += pixel.Green;
            norm++;
        }
    red /= norm;
    blue /= norm;
    green /= norm;
    res[0] = red;
    res[1] = blue;
    res[2] = green;
    return res;
}


vector<float> get_all_colors(BMP *image) {
    uint rows = image->TellWidth(), cols = image->TellHeight();
    vector<float> color_image;
    uint r1, r2, c1, c2;
    uint num_row = 8, num_col = 8;
    uint row_interval, col_interval;
    if (rows % num_row == 0)
        row_interval = round(rows / num_row);
    else
        row_interval = round(rows / num_row) + 1;
    if (cols % num_col == 0)
        col_interval = round(cols / num_col);
    else
        col_interval = round(cols / num_col) + 1;
    for (uint i = 0; i < num_row; i++)
        for (uint j = 0; j < num_col; j++) {
            r1 = i * row_interval;
            r2 = (i + 1) * row_interval;
            if (r2 >= rows) {
                r2 = rows;
            }
            c1 = j * col_interval;
            c2 = (j + 1) * col_interval;
            if (c2 >= cols) {
                c2 = cols;
            }
            if (r1 == r2 - 1 || c1 == c2 - 1) {
                continue;
            }
            auto color_cell = cell_color(image, r1, r2, c1, c2);
            color_image.insert(color_image.end(), color_cell.begin(), color_cell.end());
        }
    return color_image;
}

void norm_color(vector<float> &colors) {
    uint size = colors.size();
    for (uint i = 0; i < size; i++) {
        colors[i] /= 255;
    }
}


vector<float> make_svm_transform(vector<float> hog) {
    vector<float> result_feature;
    float L = 0.5;
    for(uint i = 0; i < hog.size(); i++){
        for(int n = -2; n < 3; n++){
            float re = 0, im = 0;
            float x = hog[i];
            if(x > 0.000001){
                re = cos(n * L * log(x)) * sqrt((2 * x) / (exp(M_PI * n * L) + exp(-M_PI * n * L)));
                im = sin(n * L * log(x)) * sqrt((2 * x) / (exp(M_PI * n * L) + exp(-M_PI * n * L)));
            }
            result_feature.push_back(re);
            result_feature.push_back(im);
        }
    }
    return result_feature;
}

// Exatract features from dataset.
void ExtractFeatures(const TDataSet &data_set, TFeatures *features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        BMP *image;
        int label;
        image = data_set[image_idx].first;
        label = data_set[image_idx].second;
        // получение серого полутонового изображения
        Matrix<float> gray_matrix = make_grayscale(image);

        // получение верт. и гориз. свертки собеля
        Matrix<float> hor_sobel = horizontal_convolution(gray_matrix),
                vert_sobel = vertical_convolution(gray_matrix);

        // вычисление градиента в каждом пикселе
        Matrix<float> module_matrix = module(hor_sobel, vert_sobel);
        Matrix<float> angle_matrix = angle(hor_sobel, vert_sobel);

        // вычисление гистограммы градиентов для всего изображения
        vector<float> hist_image = get_all_histograms(module_matrix, angle_matrix);
        norm_histogram(hist_image);

        // вычисление гистограммы градиентов для частей изображения
        // всего частей num_row x num_col
        vector<float> hist_of_all_cells;
        uint rows = module_matrix.n_rows, cols = module_matrix.n_cols;
        uint r1, r2, c1, c2;
        uint num_row = 2, num_col = 2;
        uint row_interval, col_interval;
        if (rows % num_row == 0)
            row_interval = round(rows / num_row);
        else
            row_interval = round(rows / num_row) + 1;
        if (cols % num_col == 0)
            col_interval = round(cols / num_col);
        else
            col_interval = round(cols / num_col) + 1;
        for (uint i = 0; i < num_row; i++)
            for (uint j = 0; j < num_col; j++) {
                r1 = i * row_interval;
                r2 = (i + 1) * row_interval;
                if (r2 >= rows) {
                    r2 = rows;
                }
                c1 = j * col_interval;
                c2 = (j + 1) * col_interval;
                if (c2 >= cols) {
                    c2 = cols;
                }
                Matrix<float> cell_module(r2 - r1 + 1, c2 - c1 + 1),
                        cell_angle(r2 - r1 + 1, c2 - c1 + 1);

                for (uint k = 0; k < r2 - r1; k++)
                    for (uint m = 0; m < c2 - c1; m++) {
                        cell_module(k, m) = module_matrix(k + r2 - r1 - 1, m + c2 - c1 - 1);
                        cell_angle(k, m) = angle_matrix(k + r2 - r1 - 1, m + c2 - c1 - 1);
                    }

                auto hist_current_cell = get_all_histograms(cell_module, cell_angle);
                norm_histogram(hist_current_cell);
                hist_of_all_cells.insert(hist_of_all_cells.end(), hist_current_cell.begin(),
                                         hist_current_cell.end());
            }

        // конкатенируем вектор HOG всего изображения с векторами HOG клеток
        // всего 5 векторов в итоге
        hist_image.insert(hist_image.end(), hist_of_all_cells.begin(),
                          hist_of_all_cells.end());


        // извлекаем цветовые признаки
        auto colors = get_all_colors(image);
        norm_color(colors);

        // конкатенируем цветовые признаки с вектором HOG
        hist_image.insert(hist_image.end(), colors.begin(),
                          colors.end());

        // делаем свм-преобразование
        auto result_feature = make_svm_transform(hist_image);

        // отправляем итоговый вектор в классификатор
        features->push_back(make_pair(result_feature, label));
    }
}

// Clear dataset structure
void ClearDataset(TDataSet *data_set) {
    // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
    // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string &data_file, const string &model_file) {
    // List of image file names and its labels
    TFileList file_list;
    // Structure of images and its labels
    TDataSet data_set;
    // Structure of features of images and its labels
    TFeatures features;
    // Model which would be trained
    TModel model;
    // Parameters of classifier
    TClassifierParams params;

    // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
    // Load images
    LoadImages(file_list, &data_set);
    // Extract features from images
    ExtractFeatures(data_set, &features);

    // PLACE YOUR CODE HERE
    // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
    // Train classifier
    classifier.Train(features, &model);
    // Save model to file
    model.Save(model_file);
    // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string &data_file,
                 const string &model_file,
                 const string &prediction_file) {
    // List of image file names and its labels
    TFileList file_list;
    // Structure of images and its labels
    TDataSet data_set;
    // Structure of features of images and its labels
    TFeatures features;
    // List of image labels
    TLabels labels;

    // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
    // Load images
    LoadImages(file_list, &data_set);
    // Extract features from images
    ExtractFeatures(data_set, &features);

    // Classifier
    TClassifier classifier = TClassifier(TClassifierParams());
    // Trained model
    TModel model;
    // Load model from file
    model.Load(model_file);
    // Predict images by its features using 'model' and store predictions
    // to 'labels'
    classifier.Predict(features, model, &labels);

    // Save predictions
    SavePredictions(file_list, labels, prediction_file);
    // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char **argv) {
    // Command line options parser
    ArgvParser cmd;
    // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
    // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
    // Add other options
    cmd.defineOption("data_set", "File with dataset",
                     ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
                     ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
                     ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");

    // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

    // Parse options
    int result = cmd.parse(argc, argv);

    // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

    // Get values
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

    // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
    // If we need to predict data
    if (predict) {
        // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
        // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
        // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}