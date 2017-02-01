MIT License

Copyright (c) 2017 limtiensze1974

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.




#include "stdafx.h" //<--- comment this if not using Visual Studio C++ Console Application
#include "opencv2\opencv.hpp"

using namespace cv;
using namespace std;

//Displays instructions on how to use this program
void help()
{
	cout
		<< "--------------------------------------------------------------------------" << endl
		<< "This program demonstrates the" << endl
		<< "Feature Points based Background Subtraction with Patch-based Model Update " << endl
		<< "Usage:" << endl
		<< "./main-opencv <video filename>" << endl
		<< "for example: ./main-opencv video.avi" << endl
		<< "*Note:" << endl
		<< "Edit code to change parameters if desired (see section PARAMETERS)" << endl
		<< "Press 'p' to pause video" << endl
		<< "Press 'spacebar' to manually initialize" << endl
		<< "Press 'Esc' to exit before video ends" << endl
		<< "--------------------------------------------------------------------------" << endl
		<< endl;
}

//A function to simplify Feature Map creation
Mat GetFeatureMap(Mat img) {
	Mat greyscale;
	Mat temp;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	cvtColor(img, greyscale, CV_RGB2GRAY);

	cornerHarris(greyscale, temp, 2, 9, 0);

	Mat FeatureMap;
	temp.convertTo(FeatureMap, CV_8UC1);
	threshold(FeatureMap, FeatureMap, 0, 255, CV_THRESH_BINARY);
	return FeatureMap;
}



int main(int argc, char* argv[])
{
	//Print help information
	help();

	//Check for the input parameter correctness
	if (argc != 2) {
		cerr << "Incorrect input" << endl;
		cerr << "exiting..." << endl;
		getchar();
		exit(EXIT_FAILURE);
	}


	VideoCapture cap;
	Mat frame, bgmodel;
	vector<vector<Point>> contours;

	if (!cap.open(argv[1])) {
		cout << "Video File not found" << endl;
		getchar();
		exit(EXIT_FAILURE);
	}

	int FrameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
	int FrameNo = 1;
	bool redo_decaying_threshold = false;


	//---------------------------------- PARAMETERS ---------------------------------------
	int Tau_FD = 15; 
	double Tau_SIC = 0.60; // 60% of the entire frame <Sudden illumination change response>
	double Tau_BGupdate = 0.05; // 5% of patch Area <Background model update>
	double Tau_Globalupdate = 0.02; // 2% of the entire frame <Global update>
	double Tau_GreenFeatures = 0.01; // 1% of the entire frame <Global update>


	vector<Mat> FeatureMaps(3);
	Mat CombinedFM; //Combined Feature Map
	Mat imgA;
	Mat prevA;


	cap >> bgmodel; //first frame as background model
	Mat temp = Mat(bgmodel.size(), CV_8UC1);
	temp.setTo(0);
	FeatureMaps[0] = temp;
	FeatureMaps[1] = GetFeatureMap(bgmodel);

	Mat fgMask_MOG_A; //output of MOG A
	Mat fgMask_MOG_B; //output of MOG B
	Ptr<BackgroundSubtractorMOG2> MOG_A = new BackgroundSubtractorMOG2();
	Ptr<BackgroundSubtractorMOG2> MOG_B = new BackgroundSubtractorMOG2();


	//loop
	while (true)
	{

		//Acquire next frame from the video file
		cap >> frame;
		double frameArea = (double)(frame.cols*frame.rows);
		imshow("frame", frame);

		//Frame Differencing ---------------------------------------------------------------------
		Mat temp, grey;
		absdiff(frame, bgmodel, temp);
		cvtColor(temp, grey, CV_RGB2GRAY);
		threshold(grey, imgA, Tau_FD, 255, CV_THRESH_BINARY);
		medianBlur(imgA, imgA, 3); // 3x3 median filtering, to filter out sandy noise 


		//Sudden Illumination Change Response -----------------------------------------------------
		double fgPercentage = (double)countNonZero(imgA) / frameArea;

		if (fgPercentage > Tau_SIC) //if more than 60 % of the entire scene is foreground
		{
			if (!prevA.empty())
			subtract(imgA, prevA, imgA); //subtract the mask in previous imgA from the current imgA


			MOG_B = new BackgroundSubtractorMOG2();//restart MOG_B 
			MOG_B->operator ()(frame, fgMask_MOG_B, 0);

			redo_decaying_threshold = true;
		}

		if (redo_decaying_threshold == true) //if the redo flag is set
		{
			Tau_FD = 15;
			redo_decaying_threshold = false;
		}

		//Gradually reduce the Frame Differencing threshold to 7
		if (Tau_FD > 7)
			Tau_FD--;


		//Background Update ------------------------------------------------------------------------
		FeatureMaps[2] = GetFeatureMap(frame);
		merge(FeatureMaps, CombinedFM);	//combine Feature Maps
		imshow("CombinedFM", CombinedFM);

		cvtColor(CombinedFM, temp, CV_RGB2GRAY);
		//Experiment findings:
                //Red and Green features: good for the case where expected foreground object(s) is NEAR to the camera
                //Red features only: good for the case where expected foreground object(s) is FAR from the camera
		const int RedGreen = 160;
		const int RedOnly = 120;
		int choice = RedOnly; //<---- change this if desired
		//Segment out the possible fg features from the combined Feature Map
		threshold(temp, temp, choice, 255, CV_THRESH_TOZERO_INV); 
		threshold(temp, temp, 0, 255, CV_THRESH_BINARY);
		medianBlur(temp, temp, 3); //filter out weak features



		findContours(imgA, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		Mat patch = Mat(frame.size(), CV_8UC1);  
		for (int i = 0; i < contours.size(); i++)	//Evaluate Each Contour
		{
			double patchArea = contourArea(contours[i]);
			if (patchArea / frameArea < 0.00001) 
				continue; //skip if contour is too small relatively to the image size

			patch.setTo(0);
			drawContours(patch, contours, i, Scalar(255), CV_FILLED);
			Mat Features_in_Patch;
			temp.copyTo(Features_in_Patch, patch);

			double NZ = countNonZero(Features_in_Patch);
			if (NZ / patchArea < Tau_BGupdate) //if features % lesser than 5% of the patch size
				frame.copyTo(bgmodel, patch);
		}
		imshow("bgmodel", bgmodel);

		//Frame Differencing, again, with updated background model
		absdiff(frame, bgmodel, temp);
		cvtColor(temp, grey, CV_RGB2GRAY);
		threshold(grey, imgA, Tau_FD, 255, CV_THRESH_BINARY);


		//Get Shadow / Shadow Detection ---------------------------------------------------------
		MOG_A->operator ()(frame, fgMask_MOG_A, 0.0002);
		MOG_B->operator ()(frame, fgMask_MOG_B, 0);

		Mat shadow;
		Mat shadowA;
		Mat shadowB;

		threshold(fgMask_MOG_A, shadowA, 250, 0, CV_THRESH_TOZERO_INV);
		threshold(fgMask_MOG_B, shadowB, 250, 0, CV_THRESH_TOZERO_INV);

		shadow.setTo(0);
		add(shadowA, shadowB, shadow);
		imshow("shadow", shadow);
		threshold(shadow, shadow, 50, 255, CV_THRESH_BINARY);


		// Shadow Subtraction -> Get final output
		subtract(imgA, shadow, imgA);
		medianBlur(imgA, imgA, 3); 
		imshow("Output imgA", imgA);
		imgA.copyTo(prevA); 

		//prevA will be used in the next cycle in <Frame Differencing> 
		//if sudden illumination change happen


		//Global Update -----------------------------------------------------------------------
		bool useGlobalUpdate = true; //<---- turn global update On/Off
		if (useGlobalUpdate)
		{
			cvtColor(CombinedFM, temp, CV_RGB2GRAY);
			//Segment out only the green features from the combined Feature Map
			threshold(temp, temp, 160, 255, CV_THRESH_TOZERO_INV); 
			threshold(temp, temp, 100, 255, CV_THRESH_BINARY);
			medianBlur(temp, temp, 3);

		
			double GreenCount = (double)countNonZero(temp);
			double GreenFeaturePercentage = GreenCount / frameArea;
			fgPercentage = (double)countNonZero(imgA) / frameArea;

			//if scene is empty without foreground object but large amount of green features in the scene
			if (fgPercentage < Tau_Globalupdate && GreenFeaturePercentage > Tau_GreenFeatures)
			{
				FeatureMaps[1] = GetFeatureMap(bgmodel); //update the Feature Map of background model
				MOG_A = new BackgroundSubtractorMOG2(); //restart MOG_A 
				MOG_B = new BackgroundSubtractorMOG2(); //restart MOG_B 
				prevA.setTo(0);
			}
		}



		//When end of Video is reached, breakout from loop
		FrameNo++;			
		if (FrameNo >= FrameCount) break;


		char key = waitKey(1);
		if (key == 27) //press 'Esc' to quit
			break;
		else if (key == 'p') //press 'p' to pause
			waitKey(0); 
		else if (key == ' ') //manual re-initialize
		{
			frame.copyTo(bgmodel);
			FeatureMaps[1] = GetFeatureMap(bgmodel);
			prevA.setTo(0);
			redo_decaying_threshold = true;
		}

	}

	destroyAllWindows();
	cout << "Press Enter to exit" << endl;
	getchar();
	exit(EXIT_SUCCESS);
}



