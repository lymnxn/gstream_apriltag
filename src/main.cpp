/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// #include "gstCamera.h"
// #include "glDisplay.h"
#include "videoSource.h"
#include "videoOutput.h"
#include "commandLine.h"
#include "nvAprilTags.h"
#include "imageFormat.h"

#include <signal.h>
#include <memory>
#include <string>
#include <vector>

#include <iostream>
#include <sstream>
#include <ctime>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "cuda.h"		  // NOLINT - include .h without directory
#include "cuda_runtime.h" // NOLINT - include .h without directory

bool signal_recieved = false;

void sig_handler(int signo)
{
	if (signo == SIGINT)
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

struct CailbrateData
{
	cv::Size imageSize;
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
	std::vector<std::vector<cv::Point2f>> imagePoints;
	std::string calibration_time;
	float grid_width;
	void read(const cv::FileNode &fs)
	{
		if (fs.empty())
		{
			std::cerr << "FileNode Read Error\n";
			return;
		}
		// printf("%d\n", fs.size());
		fs["calibration_time"] >> calibration_time;
		fs["cameraMatrix"] >> cameraMatrix;
		fs["image_width"] >> imageSize.width;
		fs["image_height"] >> imageSize.height;
		fs["distortion_coefficients"] >> distCoeffs;
	}
	void read(const cv::FileStorage &fs)
	{
		// calibration_time=(std::string)(fs["calibration_time"]);
		fs["calibration_time"] >> calibration_time;
		fs["camera_matrix"] >> cameraMatrix;
		fs["image_width"] >> imageSize.width;
		fs["image_height"] >> imageSize.height;
		fs["distortion_coefficients"] >> distCoeffs;
	}
	void Print()
	{
		std::cout << "calibration_time\t" << calibration_time
				  << "\nimage_width\t\t" << imageSize.width
				  << "\nimage_height\t\t" << imageSize.height << std::endl;
		std::cout << cameraMatrix << std::endl;
		std::cout << distCoeffs << std::endl;
	}
};

struct AprilTagsInfo
{
	float tag_edge_size_;
	size_t max_tags_;
};

struct AprilTagsImpl
{
	// Handle used to interface with the stereo library.
	nvAprilTagsHandle april_tags_handle = nullptr;

	// Camera intrinsics
	nvAprilTagsCameraIntrinsics_t cam_intrinsics;

	// Output vector of detected Tags
	std::vector<nvAprilTagsID_t> tags;

	// CUDA stream
	cudaStream_t main_stream = {};

	// CUDA buffers to store the input image.
	nvAprilTagsImageInput_t input_image;

	// CUDA memory buffer container for RGBA images.
	char *input_image_buffer = nullptr;

	// Size of image buffer
	size_t input_image_buffer_size = 0;

	void initialize(const AprilTagsInfo &node, const CailbrateData &cali, const uint32_t width,
					const uint32_t height, const size_t image_buffer_size,
					const size_t pitch_bytes)
	{
		// std::assert(april_tags_handle == nullptr && "Already initialized.");

		// Get camera intrinsics
		// const double * k = msg_ci->k.data();
		// const float fx = static_cast<float>(k[0]);
		// const float fy = static_cast<float>(k[4]);
		// const float cx = static_cast<float>(k[2]);
		// const float cy = static_cast<float>(k[5]);
		// std::cout<<"A\n";
		const float fx = (float)*(double *)(cali.cameraMatrix.row(0).col(0).data);
		const float fy = (float)*(double *)(cali.cameraMatrix.row(1).col(1).data);
		const float cx = (float)*(double *)(cali.cameraMatrix.row(0).col(2).data);
		const float cy = (float)*(double *)(cali.cameraMatrix.row(1).col(2).data);
		cam_intrinsics = {fx, fy, cx, cy};
		// std::cout<<"B\n";
		// Create AprilTags detector instance and get handle
		const int error = nvCreateAprilTagsDetector(
			&april_tags_handle, width, height, nvAprilTagsFamily::NVAT_TAG36H11,
			&cam_intrinsics, node.tag_edge_size_);
		if (error != 0)
		{
			throw std::runtime_error(
				"Failed to create NV April Tags detector (error code " +
				std::to_string(error) + ")");
		}
		// std::cout<<"C\n";

		// Create stream for detection
		cudaStreamCreate(&main_stream);
		// std::cout<<"D\n";
		// Allocate the output vector to contain detected AprilTags.
		tags.resize(node.max_tags_);
		// std::cout<<"E\n";
		// Setup input image CUDA buffer.
		// const cudaError_t cuda_error =
		// 	cudaMalloc(&input_image_buffer, image_buffer_size);
		// if (cuda_error != cudaSuccess)
		// {
		// 	throw std::runtime_error(
		// 		"Could not allocate CUDA memory (error code " +
		// 		std::to_string(cuda_error) + ")");
		// }
		// std::cout<<"F\n";
		// Setup input image.
		input_image_buffer_size = image_buffer_size;
		input_image.width = width;
		input_image.height = height;
		input_image.dev_ptr = reinterpret_cast<uchar4 *>(input_image_buffer);
		input_image.pitch = pitch_bytes;

		// std::cout<<"G\n";
	}

	~AprilTagsImpl()
	{
		if (april_tags_handle != nullptr)
		{
			cudaStreamDestroy(main_stream);
			nvAprilTagsDestroy(april_tags_handle);
			cudaFree(input_image_buffer);
		}
	}
};

int main(int argc, char **argv)
{
	commandLine cmdLine(argc, argv);
	if (!cmdLine.GetFlag("file"))
	{
		printf("Error input, please use:\" --file \"\n");
		return -1;
	}
	const char *inputstr = cmdLine.GetString("file");
	// printf("%X\n", (unsigned int)inputstr);
	if (inputstr == NULL)
	{
		cmdLine.Print();
		std::cout << "Input Error" << std::endl;
		return -1;
	}
	// printf("%s\n", inputstr);
	std::string inputCameraFile(inputstr);
	std::cout << "Input File: \"" << inputCameraFile << "\"" << std::endl;
	cv::FileStorage fs(inputCameraFile, cv::FileStorage::READ);
	CailbrateData caildata;
	AprilTagsInfo aprilti{10.0, 20};
	std::unique_ptr<AprilTagsImpl> impl_(std::make_unique<AprilTagsImpl>());
	if (!fs.isOpened())
	{
		std::cout << "Could not open the configuration file: \"" << inputCameraFile << "\"" << std::endl;
		return -1;
	}
	else
	{
		std::cout << "Open the configuration file: \"" << inputCameraFile << "\" Successfully!" << std::endl;
		caildata.read(fs);
		caildata.Print();
		fs.release();
	}
	// std::cout<<caildata.cameraMatrix.row(0).col(0)<<"\t"<<*(double*)(caildata.cameraMatrix.row(0).col(0).data)<<std::endl;

	/*
	 * attach signal handler
	 */
	if (signal(SIGINT, sig_handler) == SIG_ERR)
		printf("\ncan't catch SIGINT\n");

	/*
	 * create the camera device
	//  */
	// gstCamera *camera = gstCamera::Create(cmdLine.GetInt("width", gstCamera::DefaultWidth),
	// 									  cmdLine.GetInt("height", gstCamera::DefaultHeight),
	// 									  cmdLine.GetString("camera"));

	// if (!camera)
	// {
	// 	printf("\ncamera-viewer:  failed to initialize camera device\n");
	// 	return 0;
	// }

	// printf("\ncamera-viewer:  successfully initialized camera device (%ux%u)\n", camera->GetWidth(), camera->GetHeight());

	// /*
	//  * create openGL window
	//  */
	// glDisplay *display = glDisplay::Create("", camera->GetWidth() + 100, camera->GetHeight() + 100);

	// if (!display)
	// 	printf("camera-viewer:  failed to create openGL display\n");

	// /*
	//  * start streaming
	//  */
	// if (!camera->Open())
	// {
	// 	printf("camera-viewer:  failed to open camera for streaming\n");
	// 	return 0;
	// }

	// printf("camera-viewer:  camera open for streaming\n");
	videoOptions opt;
	if (!cmdLine.GetFlag("width"))
	{
		printf("Not Find width, use 960\n");
		opt.width = 1280;
	}
	else
	{
		opt.width = cmdLine.GetInt("width");
	}
	if (!cmdLine.GetFlag("height"))
	{
		printf("Not Find height, use 720\n");
		opt.height = 720;
	}
	else
	{
		opt.height = cmdLine.GetInt("height");
	}
	if (!cmdLine.GetFlag("fps"))
	{
		printf("Not Find fps, use 60\n");
		opt.frameRate = 60;
	}
	else
	{
		opt.frameRate = cmdLine.GetInt("fps");
	}
	opt.zeroCopy = false;
	std::cout << "Camera Info:\nWidth = " << opt.width << "\nHeight = " << opt.height << "\nFrameRate = " << opt.frameRate << "\n"
			  << opt.FlipMethodToStr << std::endl;
	videoSource *input = videoSource::Create("csi://0", opt);

	if (!input)
	{
		std::cerr << "Error: Failed to create input stream" << std::endl;
		exit(-1);
	}

	// create output stream
	videoOutput *output = videoOutput::Create("display://0");
	if (!output)
	{
		std::cerr << "Error: Failed to create output stream" << std::endl;
		delete input;
		exit(-2);
	}

	/*
	 * processing loop
	 */
	unsigned int framecnt = 0;
	while (!signal_recieved)
	{
		// capture latest image
		uchar4 *imgRGBA = NULL;

		// if (!camera->CaptureRGBA8(&imgRGBA, 5000, true))
		// 	printf("camera-viewer:  failed to capture RGBA image\n");
		if (!input->Capture(&imgRGBA, 1000))
		{
			if (!input->IsStreaming())
				break;

			std::cerr << "failed to capture next frame\n";
			continue;
		}
		// cv::cuda::GpuMat img_rgba8(camera->GetWidth(), camera->GetHeight(), CV_8UC4, imgRGBA);

		// std::cout<<"Step" << img_rgba8.step<< std::endl;	//	2880
		// std::cout<<"Hello Moto"<<img_rgba8.elemSize()<<"\t"<<img_rgba8.rows<<"\t"<<img_rgba8.cols<<std::endl;

		// cv::cuda::GpuMat img(input->GetWidth(), input->GetHeight(), CV_8UC4, imgRGBA);
		// cv::Mat img_rgba8=img.download
		if (impl_->april_tags_handle == nullptr)
		{
			cv::cuda::GpuMat img_rgba8(input->GetHeight(), input->GetWidth(), CV_8UC4, imgRGBA);
			impl_->initialize(aprilti, caildata, input->GetWidth(), input->GetHeight(), img_rgba8.size().width * img_rgba8.size().height * img_rgba8.elemSize(), img_rgba8.step1());
		}
		// cv::Mat test(input->GetHeight(), input->GetWidth(), CV_8UC4);
		// std::cout<<"Step = "<<img_rgba8.step<<"\t"<<img_rgba8.step1()<<"\n"<< "Size = "<< img_rgba8.size() <<"Test step: "<<test.step<<"\n"<<"Element Size = "<<img_rgba8.elemSize()<<"\n";
		// std::cout<<aprilti.max_tags_<<"\t"<<aprilti.tag_edge_size_<<std::endl;//10
		// std::cout<<impl_->cam_intrinsics.fx<<"\t"<<impl_->cam_intrinsics.fy<<"\n"
		// 		<<impl_->cam_intrinsics.cx<<"\t"<<impl_->cam_intrinsics.cy<<std::endl;

		// const cudaError_t cuda_error =
		// 	cudaMemcpy(
		// 		impl_->input_image_buffer, img_rgba8.ptr(),
		// 		impl_->input_image_buffer_size, cudaMemcpyHostToDevice);
		// if (cuda_error != cudaSuccess)
		// {
		// 	std::cerr << "Could not memcpy to device CUDA memory (error code " << std::to_string(cuda_error) << ")\n";
		// 	return -1;
		// }
		// impl_->input_image.dev_ptr=imgRGBA;
		impl_->input_image_buffer = (char *)imgRGBA;
		impl_->input_image.dev_ptr = reinterpret_cast<uchar4 *>(imgRGBA);
		uint32_t num_detections;
		const int error = nvAprilTagsDetect(
			impl_->april_tags_handle, &(impl_->input_image), impl_->tags.data(),
			&num_detections, aprilti.max_tags_, impl_->main_stream);
		if (error != 0)
		{
			std::cerr << "Failed to run AprilTags detector (error code " << error << ")\n";
			return -1;
		}

		// std::cout<<"Apriltag cnt = "<< num_detections<<std::endl;
		// update display
		// if (display != NULL)
		// {
		// 	display->RenderOnce(imgRGBA, camera->GetWidth(), camera->GetHeight(), IMAGE_RGBA8, 5.0f, 30.0f, false);

		// 	// update status bar
		// 	char str[256];
		// 	sprintf(str, "Camera Viewer (%ux%u) | %.0f FPS", camera->GetWidth(), camera->GetHeight(), display->GetFPS());
		// 	display->SetTitle(str);

		// 	// check if the user quit
		// 	if (display->IsClosed())
		// 		signal_recieved = true;
		// }
		if (output != NULL)
		{
			static int cnt = 0;
			static float sum = 0;
			output->Render(imgRGBA, input->GetWidth(), input->GetHeight());
			sum += output->GetFrameRate();
			// update status bar
			if (cnt == 9)
			{
				static char str[256];
				sprintf(str, "Camera Viewer (%ux%u) | %.0f FPS %d", input->GetWidth(), input->GetHeight(), sum /= 10, num_detections);
				output->SetStatus(str);
				if (num_detections)
				{
					// std::cout << "Frame : " << framecnt << "\t Apriltag num : " << num_detections << std::endl;
					for (int i = 0; i < num_detections; ++i)
					{
						if(impl_->tags[i].id!=2)
							continue;
						// std::cout << impl_->tags[i].id << "\n";
						for (int j = 0; j < 3; ++j)
							std::cout << impl_->tags[i].translation[j] << "\t\t";
						std::cout << std::endl;
						for(int j=0;j<4;++j)
						{
							std::cout<<impl_->tags[i].corners[j].x<<" \t "<<impl_->tags[i].corners[j].y<<std::endl;
						}
					}
				}
				sum = 0;
				cnt = 0;
			}
			else
				++cnt;

			// check if the user quit
			if (!output->IsStreaming())
				signal_recieved = true;
		}
	}

	/*
	 * destroy resources
	 */
	printf("\ncamera-viewer:  shutting down...\n");

	// SAFE_DELETE(camera);
	// SAFE_DELETE(display);
	SAFE_DELETE(input);
	SAFE_DELETE(output);

	printf("camera-viewer:  shutdown complete.\n");
	return 0;
}
