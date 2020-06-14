
// FaceRecognitionDlg.cpp: 实现文件
//

#include "pch.h"
#include "framework.h"
#include "FaceRecognition.h"
#include "FaceRecognitionDlg.h"
#include "afxdialogex.h"



#include <thread>
#include<opencv2/contrib/contrib.hpp>
#include<numeric>
#ifdef _DEBUG
#define new DEBUG_NEW
#endif

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);
// 用于应用程序“关于”菜单项的 CAboutDlg 对话框
const static std::string request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect";
static std::string faceDetect_result;
const std::string token = "24.477bf13b75acb11f1c4483baaf806464.2592000.1594171970.282335-20282704";
int m_int;
int bilateralFilterVal = 30;  // 双边模糊系数
bool flag = false;
Json::CharReaderBuilder readerBuilder;
Json::Value face_list, result, mask, root, face_list_2, face_shape, emotion,gender;
JSONCPP_STRING errs;
std::unique_ptr<Json::CharReader> const jsonReader(readerBuilder.newCharReader());
/**
 * curl发送http请求调用的回调函数，回调函数中对返回的json格式的body进行了解析，解析结果储存在全局的静态变量当中
 * @param 参数定义见libcurl文档
 * @return 返回值定义见libcurl文档
 */


std::string codes,res_jason;
static size_t callback(void *ptr, size_t size, size_t nmemb, void *stream) {
	// 获取到的body存放在ptr中，先将其转换为string格式
	faceDetect_result = std::string((char *)ptr, size * nmemb);
	return size * nmemb;
}
/**
 * 人脸检测与属性分析
 * @return 调用成功返回0，发生错误返回其他错误码
 */

void whiteFace(Mat& matSelfPhoto, int alpha, int beta)
{
	for (int y = 0; y < matSelfPhoto.rows; y++)
	{
		for (int x = 0; x < matSelfPhoto.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				matSelfPhoto.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(alpha*(matSelfPhoto.at<Vec3b>(y, x)[c]) + beta);
			}
		}
	}
}

int faceDetect(std::string &json_result, const std::string &access_token, std::string &src) {
	std::string url = request_url + "?access_token=" + access_token;
	CURL *curl = NULL;
	CURLcode result_code;
	int is_success;
	curl = curl_easy_init();

	if (curl) {

		curl_easy_setopt(curl, CURLOPT_URL, url.data());
		curl_easy_setopt(curl, CURLOPT_POST, 1);
		curl_slist *headers = NULL;
		headers = curl_slist_append(headers, "Content-Type:application/json;");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &faceDetect_result);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, callback);


		string temp = "{\"image\":\"";
		temp.append(src);
		temp.append("\",\"image_type\":\"BASE64\",\"face_field\":\"age,beauty,emotion,face_shape,gender,mask\"}");

		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, temp.data());
		result_code = curl_easy_perform(curl);
		cout << "res_code:" << result_code << endl;
		if (result_code != CURLE_OK) {
			fprintf(stderr, "curl_easy_perform() failed: %s\n",
				curl_easy_strerror(result_code));
			is_success = 1;
			return is_success;
		}

		json_result = faceDetect_result;
		curl_easy_cleanup(curl);
		is_success = 0;
	}
	else {
		fprintf(stderr, "curl_easy_init() failed.");
		is_success = 1;
	}
	return is_success;
}

//imgType 包括png bmp jpg jpeg等opencv能够进行编码解码的文件
static std::string base64Encode(const unsigned char* Data, int DataByte) {
	//编码表
	const char EncodeTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	//返回值
	std::string strEncode;
	unsigned char Tmp[4] = { 0 };
	int LineLength = 0;
	for (int i = 0; i < (int)(DataByte / 3); i++) {
		Tmp[1] = *Data++;
		Tmp[2] = *Data++;
		Tmp[3] = *Data++;
		strEncode += EncodeTable[Tmp[1] >> 2];
		strEncode += EncodeTable[((Tmp[1] << 4) | (Tmp[2] >> 4)) & 0x3F];
		strEncode += EncodeTable[((Tmp[2] << 2) | (Tmp[3] >> 6)) & 0x3F];
		strEncode += EncodeTable[Tmp[3] & 0x3F];
		//if (LineLength += 4, LineLength == 76) { strEncode += "\r\n"; LineLength = 0; }
	}
	//对剩余数据进行编码
	int Mod = DataByte % 3;
	if (Mod == 1) {
		Tmp[1] = *Data++;
		strEncode += EncodeTable[(Tmp[1] & 0xFC) >> 2];
		strEncode += EncodeTable[((Tmp[1] & 0x03) << 4)];
		strEncode += "==";
	}
	else if (Mod == 2) {
		Tmp[1] = *Data++;
		Tmp[2] = *Data++;
		strEncode += EncodeTable[(Tmp[1] & 0xFC) >> 2];
		strEncode += EncodeTable[((Tmp[1] & 0x03) << 4) | ((Tmp[2] & 0xF0) >> 4)];
		strEncode += EncodeTable[((Tmp[2] & 0x0F) << 2)];
		strEncode += "=";
	}


	return strEncode;
}

static std::string Mat2Base64(const cv::Mat &img, std::string imgType) {
	//Mat转base64
	std::string img_data;
	std::vector<uchar> vecImg;
	std::vector<int> vecCompression_params;
	vecCompression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	vecCompression_params.push_back(90);
	imgType = "." + imgType;
	cv::imencode(imgType, img, vecImg, vecCompression_params);
	img_data = base64Encode(vecImg.data(), vecImg.size());
	return img_data;
}
cv::Mat II;
std::vector<matrix<float, 0, 1>> vec;        //定义一个向量组，用于存放每一个人脸的编码；
float vec_error[30];                         //定义一个浮点型的数组，用于存放一个人脸编码与人脸库的每一个人脸编码的差值；
matrix<rgb_pixel> img, img1, img3;            //定义dlib型图片，彩色
std::vector<string> fileNames;

string  xingbie;

std::vector<String> ageLabels() {
	std::vector<String> ages;
	ages.push_back("0-2");
	ages.push_back("4 - 6");
	ages.push_back("8 - 13");
	ages.push_back("15 - 20");
	ages.push_back("25 - 32");
	ages.push_back("38 - 43");
	ages.push_back("48 - 53");
	ages.push_back("60-");
	return ages;
}

string predict_age(Net &net, Mat &image) {
	// 输入
	Mat blob = blobFromImage(image, 1.0, Size(227, 227));
	net.setInput(blob, "data");
	// 预测分类
	Mat prob = net.forward("prob");
	Mat probMat = prob.reshape(1, 1);
	Point classNum;
	double classProb;

	std::vector<String> ages = ageLabels();
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNum);
	
	int classidx = classNum.x;
	//putText(image, format("age:%s", ages.at(classidx).c_str()), Point(2, 10), FONT_HERSHEY_PLAIN, 0.8, Scalar(0, 0, 255), 1);
	return  ages.at(classidx).c_str();
}

string predict_gender(Net &net, Mat &image) {
	// 输入
	Mat blob = blobFromImage(image, 1.0, Size(227, 227));
	net.setInput(blob, "data");
	// 预测分类
	Mat prob = net.forward("prob");
	Mat probMat = prob.reshape(1, 1);
	//putText(image, format("gender:%s", (probMat.at<float>(0, 0) > probMat.at<float>(0, 1) ? "M" : "F")),
	//	Point(2, 20), FONT_HERSHEY_PLAIN, 0.8, Scalar(0, 0, 255), 1);
	return probMat.at<float>(0, 0) > probMat.at<float>(0, 1) ? "Male" : "Female";
	
}

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CFaceRecognitionDlg 对话框



CFaceRecognitionDlg::CFaceRecognitionDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_FACERECOGNITION_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CFaceRecognitionDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);

	//DDX_Control(pDX, IDC_SLIDER1, m_slider);
}

BEGIN_MESSAGE_MAP(CFaceRecognitionDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &CFaceRecognitionDlg::OnBnClickedOk)
	ON_WM_TIMER()
	ON_BN_CLICKED(IDCANCEL, &CFaceRecognitionDlg::OnBnClickedCancel)
	ON_BN_CLICKED(IDC_BUTTON1, &CFaceRecognitionDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CFaceRecognitionDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CFaceRecognitionDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CFaceRecognitionDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CFaceRecognitionDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CFaceRecognitionDlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &CFaceRecognitionDlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &CFaceRecognitionDlg::OnBnClickedButton8)
	ON_WM_CTLCOLOR()
	ON_STN_CLICKED(IDC_IMAGE_SHOW, &CFaceRecognitionDlg::OnStnClickedImageShow)
	ON_WM_HSCROLL()
	ON_EN_CHANGE(IDC_EDIT1, &CFaceRecognitionDlg::OnEnChangeEdit1)
END_MESSAGE_MAP()


// CFaceRecognitionDlg 消息处理程序

BOOL CFaceRecognitionDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。
	m_slider.SetRange(0, 4);//设置滑动范围为1到20
	m_slider.SetTicFreq(1);//每1个单位画一刻度
	m_slider.SetPos(0);//设置滑块初始位置为1 
	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	/*
	CWnd  *pWnd1 = GetDlgItem(IDC_PIC);//CWnd是MFC窗口类的基类,提供了微软基础类库中所有窗口类的基本功能。
	pWnd1->GetClientRect(&m_rect);//GetClientRect为获得控件相自身的坐标大小
	namedWindow("src1", WINDOW_AUTOSIZE);//设置窗口名
	HWND hWndl = (HWND)cvGetWindowHandle("src1");//hWnd 表示窗口句柄,获取窗口句柄
	HWND hParent1 = ::GetParent(hWndl);//GetParent函数一个指定子窗口的父窗口句柄
	::SetParent(hWndl, GetDlgItem(IDC_PIC)->m_hWnd);
	::ShowWindow(hParent1, SW_HIDE);
	Mat srcImg = imread("婉儿.jpg");
	resize(srcImg, srcImg, Size(m_rect.Width(), m_rect.Height()));
	imshow("src1", srcImg);
	*/
	//----------------------------【自定义代码处】----

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CFaceRecognitionDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CFaceRecognitionDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CFaceRecognitionDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}





void CFaceRecognitionDlg::OnBnClickedOk()
{
	// TODO: 在此添加控件通知处理程序代码
	
	//MessageBoxA(NULL, "Hello, Windows!", "hello", MB_OK);
	//MessageBox.Show("确定要删除" + xx + "吗？", "警告！！！", MessageBoxButtons.OKCancel);
	OutputDebugString(_T("*******************************************************************************************\n"));
	init();
	
	SetTimer(1, 350, NULL);
	
	OutputDebugString(_T("------------------------------------------------------------------------------------------------------------------------------------------------\n"));
	//CDialogEx::OnOK();
}

void CFaceRecognitionDlg::init()
{
	namedWindow("view", WINDOW_AUTOSIZE);         //button部分
	HWND hWnd = (HWND)cvGetWindowHandle("view");
	HWND hParent = ::GetParent(hWnd);
	::SetParent(hWnd, GetDlgItem(IDC_STATIC)->m_hWnd);
	::ShowWindow(hParent, SW_HIDE);
	

	CRect rect;
	GetDlgItem(IDC_STATIC)->GetClientRect(&rect);
	Rect dst(rect.left, rect.top, rect.right, rect.bottom);
	Mat result = imread("logo.jpg");
	cv::resize(result, result, cv::Size(rect.Width(), rect.Height()));

	imshow("view", result);

	//OutputDebugString(_T("success!!!!\n"));
	if (!m_cvCap.isOpened())
	{
		m_cvCap = 0; // 打开摄像头
		signal_point = 0;
		// CDC是MFC的DC的一个类
		m_pDC = GetDlgItem(IDC_IMAGE_SHOW)->GetDC();

		// 获取设备上下文的句柄
		m_hDC = m_pDC->GetSafeHdc();

		// 获取绘制区域
		GetDlgItem(IDC_IMAGE_SHOW)->GetClientRect(&m_rect);
		String modelConfiguration = "D:/opencv/opencv/sources/samples/dnn/face_detector/deploy.prototxt";
		String modelBinary = "D:/opencv/opencv/sources/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel";
		String age_model_deploy = "D:/opencv/opencv/sources/samples/dnn/face_detector/models/age_deploy.prototxt";
		String age_model = "D:/opencv/opencv/sources/samples/dnn/face_detector/models/age_net.caffemodel";
		String gender_model_deploy = "D:/opencv/opencv/sources/samples/dnn/face_detector/models/gender_deploy.prototxt";
		String gender_model = "D:/opencv/opencv/sources/samples/dnn/face_detector/models/gender_net.caffemodel";
		try
		{
			faceNet = readNetFromCaffe(modelConfiguration, modelBinary);
			ageNet = readNetFromCaffe(age_model_deploy, age_model);
			genderNet = readNetFromCaffe(gender_model_deploy,gender_model);
			detector = get_frontal_face_detector();
			deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
			deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
			
			Directory dir;
			string dir_path = "G:/pic";
			fileNames = dir.GetListFiles(dir_path, "*.jpg", false);;//统计文件夹里jpg格式文件的个数，并将每个文件的名字保存
			for (int k = 0; k < fileNames.size(); k++)
			{
				string fileFullName = dir_path + "//" + fileNames[k];//图片地址+文件名
				load_image(img, fileFullName);//load picture      //加载图片
											  // Display the raw image on the screen
				std::vector<dlib::rectangle> dets = detector(img);  //用dlib自带的人脸检测器检测人脸，然后将人脸位置大小信息存放到dets中
				std::vector<matrix<rgb_pixel>> faces;//定义存放截取人脸数据组

				auto shape = pose_model(img, dets[0]);
				matrix<rgb_pixel> face_chip;
				extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);//截取人脸部分，并将大小调为150*150
				faces.push_back(move(face_chip));

				std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
				vec.push_back(face_descriptors[0]);
			}
			CString a(fileNames.at(0).c_str());
			OutputDebugString(a+"##################################################################\n");
			
			OutputDebugString(_T("init  is  success!!!!\n"));
		}
		catch (const std::exception&)
		{
			OutputDebugString(_T("FAILED!!!!"));
		}
		
	   
	}
	


}

void CFaceRecognitionDlg::unInit()
{
	if (m_cvCap.isOpened())
	{
		m_cvCap.release();
		m_cvCap = -1;

		if (m_pDC)
		{
			ReleaseDC(m_pDC);
			m_pDC = NULL;
		}
	}
}

int CFaceRecognitionDlg::drawPicToHDC(IplImage * pImg)
{
	
	if (!pImg)
	{
		OutputDebugString(_T("pImg is NULL!\n"));
		return -1;
	}

	m_bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	m_bmpInfo.bmiHeader.biHeight = pImg->height;
	m_bmpInfo.bmiHeader.biWidth = pImg->width;
	m_bmpInfo.bmiHeader.biPlanes = 1;
	m_bmpInfo.bmiHeader.biBitCount = 24;
	m_bmpInfo.bmiHeader.biCompression = BI_RGB;
	m_bmpInfo.bmiHeader.biSizeImage = pImg->imageSize;
	m_bmpInfo.bmiHeader.biXPelsPerMeter = 0;
	m_bmpInfo.bmiHeader.biYPelsPerMeter = 0;
	m_bmpInfo.bmiHeader.biClrUsed = 0;
	m_bmpInfo.bmiHeader.biClrImportant = 0;

	// 该函数可以设置指定输出设备环境中的位图拉伸模式
	SetStretchBltMode(m_hDC, COLORONCOLOR);
	
	::StretchDIBits(
		m_hDC,
		// 默认绘制原点为左上角，X方向向右为正，Y方向向下为正
		// 加上负号表明X或者Y方向取反
		m_rect.Width(), m_rect.Height(), -m_rect.Width(), -m_rect.Height(),
		0, 0, m_bmpInfo.bmiHeader.biWidth, m_bmpInfo.bmiHeader.biHeight,
		pImg->imageData, (PBITMAPINFO)&m_bmpInfo, DIB_RGB_COLORS, SRCCOPY);

	return 0;
}


Mat poolMat;
Point mpoint;
void MultilPoolFunction(Mat frame,int canshu) {
	string text;
	codes = Mat2Base64(frame, "jpg");

	faceDetect(res_jason, token, codes);

	flag = jsonReader->parse(res_jason.c_str(), res_jason.c_str() + res_jason.length(), &root, &errs);
	if (!flag || !errs.empty()) {
		OutputDebugString(_T("parseJson err"));

	}

	result = root["result"];
	face_list = result["face_list"];
	face_list_2 = face_list[0];
	if (canshu == -1) {//情绪测试
		emotion = face_list_2["emotion"];
		text = emotion["type"].asString();
	}
	else if (canshu == 0) {//颜值打分
		string beauty = face_list_2["beauty"].asString();
		text = "Score " + beauty;
	}
	else if (canshu == 1) {//口罩检测
		mask = face_list_2["mask"];
		int mask_res = mask["type"].asInt();
		if (mask_res == 0) {
			text = "Missing mask";
		}
		else {
			text = "Wearing mask";
		}
	}
	else if (canshu == 2) {//脸型判断
		face_shape = face_list_2["face_shape"];
		string shape_face = face_shape["type"].asString();
		text = "Shape " + shape_face;
	}
	else {//预测年龄
		string age = face_list_2["age"].asString();
		gender = face_list_2["gender"];
		xingbie = gender["type"].asString();
		text = "Age " + age+" Sex "+xingbie;
		
		// Mat res = imread("婉儿.jpg");
		
	}
	
	putText(frame, text, Point(50, 60), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 23, 0), 4, 8);
	poolMat = frame;
	
};
void CFaceRecognitionDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值
	Mat frame;
	m_cvCap >> frame;
	if (m_int == 1) {
		whiteFace(frame, 1.1, 68);
	}
	if (m_int == 2) {
		GaussianBlur(frame, frame, Size(9, 9), 0, 0);
	}
	if (m_int == 3) {
		Mat t;
		bilateralFilter(frame, t, bilateralFilterVal, // 整体磨皮
			bilateralFilterVal * 2, bilateralFilterVal / 2);
		frame = t;
	}
	if (m_int == 4) {
		Mat matResult;
		bilateralFilter(frame, matResult, bilateralFilterVal, // 整体磨皮
			bilateralFilterVal * 2, bilateralFilterVal / 2);

		Mat matFinal;

		// 图像增强，使用非锐化掩蔽（Unsharpening Mask）方案。
		cv::GaussianBlur(matResult, matFinal, cv::Size(0, 0), 9);
		cv::addWeighted(matResult, 1.5, matFinal, -0.5, 0, matFinal);

		frame = matFinal;
		
	}



	if (signal_point == 1) {
		try
		{
			
			
			cv_image<bgr_pixel> img(frame);
			// Detect faces 
			std::vector<dlib::rectangle> faces = detector(img);
			// Find the pose of each face.
			std::vector<full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(pose_model(img, faces[i]));

			if (!shapes.empty()) {
				for (int j = 0; j < shapes.size(); j++) {
					for (int i = 0; i < 68; i++) {
						circle(frame, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 3, cv::Scalar(0, 0, 255), -1);
						//	shapes[0].part(i).x();//68个

					}

				}
				
			}
			//imshow("temp", frame);
			
			

		}
		catch (serialization_error& e)
		{
			cout << "You need dlib's default face landmarking model file to run this example." << endl;
			cout << "You can get it from the following URL: " << endl;
			cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
			cout << endl << e.what() << endl;
		}
		catch (exception& e)
		{
			cout << e.what() << endl;
		}

	

		

	}
	else if (signal_point == 2) {
		try
		{


			cv_image<bgr_pixel> src(frame);
			std::vector<matrix<rgb_pixel>> faces_test;
			for (auto face_test : detector(src))
			{
				auto shape_test =pose_model(src, face_test);
				matrix<rgb_pixel> face_chip_test;
				extract_image_chip(src, get_face_chip_details(shape_test, 150, 0.25), face_chip_test);
				faces_test.push_back(move(face_chip_test));
				// Also put some boxes on the faces so we can see that the detector is finding
				// them.

			}
			//===========
			
			//=========
			std::vector<dlib::rectangle> dets_test = detector(src);
			std::vector<matrix<float, 0, 1>> face_test_descriptors = net(faces_test);
			std::vector<sample_pair> edges;
			for (size_t i = 0; i < face_test_descriptors.size(); ++i)                 //比对，识别
			{
				size_t m = 100;
				float error_min = 100.0;
				for (size_t j = 0; j < vec.size(); ++j)
				{
							
					vec_error[j] = (double)length(face_test_descriptors[i] - vec[j]);
					//cout << "The error of two picture is:" << vec_error[j] << endl;

					//if (length(face_descriptors[i] - face_descriptors[j]) < 0.6)
					if (vec_error[j] < error_min)
					{
						error_min = vec_error[j];
						m = j;

					}

				}
				//cout << "min error of two face:" << error_min << endl;
				II = dlib::toMat(src);//½«dlibÍ¼Ïñ×ªµ½opencv
				std::string text = "Unknown";
				if ((error_min < 0.4) && (m <= 27)) {
					String tempStr= fileNames[m];
					int pos=tempStr.find_first_of(".");
					//char * tempStr = strtok(fileNames[m],".");
					text = tempStr.substr(0,pos);  //通过m定位文件，得到文件名
				}
					


				int font_face = cv::FONT_HERSHEY_COMPLEX;
				double font_scale = 1;
				int thickness = 2;
				int baseline;
				//获取文本框的长宽
				cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

				//将文本框居中绘制
				cv::Point origin;


				cv::rectangle(II, cv::Rect(dets_test[i].left(), dets_test[i].top(), dets_test[i].width(), dets_test[i].width()), cv::Scalar(0, 0, 255), 1, 1, 0);//画矩形框
				origin.x = dets_test[i].left();
				origin.y = dets_test[i].top();
				cv::putText(II, text, origin, font_face, font_scale, cv::Scalar(255, 0, 0), thickness, 2, 0);//给图片加文字

				frame = II;
			}
			
			
			
			

				//image_window win4(img4);
				//std::vector<matrix<rgb_pixel>> faces_test;



		//----------------test


			


		}
		catch (serialization_error& e)
		{
			cout << "You need dlib's default face landmarking model file to run this example." << endl;
			cout << "You can get it from the following URL: " << endl;
			cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
			cout << endl << e.what() << endl;
		}
		catch (exception& e)
		{
			cout << e.what() << endl;
		}


	}
	else if (signal_point == 3) {
	MultilPoolFunction(frame, -1);
	/*
	try
	{


		cv_image<bgr_pixel> img(frame);
		// Detect faces 
		std::vector<dlib::rectangle> faces = detector(img);
		
		// Find the pose of each face.
		std::vector<full_object_detection> shapes;
		for (unsigned long i = 0; i < faces.size(); ++i)
			shapes.push_back(pose_model(img, faces[i]));

		for (int i = 0; i < 68; i++) {
			circle(frame, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 3, cv::Scalar(0, 0, 255), -1);
			//	shapes[0].part(i).x();//68个

		}
		
		//imshow("tttt",toMat(shapes[0]));
		if (!shapes.empty()) {
			for (int j = 0; j < shapes.size(); j++) {
			    //对每一张脸的表情进行预测分析
				
				string text="";
				auto face_width = faces[j].right() - faces[j].left();
				auto face_height= faces[j].top() - faces[j].bottom();
				auto mouth_width = (shapes[j].part(54).x() - shapes[j].part(48).x()) / face_width;
				auto mouth_height= ((shapes[j].part(66).y() - shapes[j].part(62).y())/1.0) / face_width;
				//zui.push_back(mouth_height);
				
				//接下来进行的是线性回归（根据点来拟合出一条直线）
			
				std::vector<cv::Point> points;
				cv::Vec4f res;
				for (int  i = 17; i <=21; i++)
				{
					points.push_back(cv::Point(shapes[j].part(i).x(), shapes[j].part(i).y()));
				
				}
				fitLine(points, res, CV_DIST_HUBER, 0, 0.01, 0.01);
				double res_xielv = -(res[1] / res[0]);
				//xielv.push_back(res_xielv);
				//CString d(res_xielv);
				
				

				auto eye_sum = (shapes[j].part(41).y() - shapes[j].part(37).y() + shapes[j].part(40).y() - shapes[j].part(38).y() +
					shapes[j].part(47).y() - shapes[j].part(43).y() + shapes[j].part(46).y() - shapes[j].part(44).y());
				auto eye_hight = (eye_sum / 4.0) / face_width;
				
				
				//yan.push_back(eye_hight);
				if ((mouth_height >= 0.101)) {
					if (eye_hight >= 0.035) { text = "amazing"; }
					else { text = "happy"; }
				}
				// 没有张嘴，可能是正常和生气
				else {
					if (res_xielv <= -0.1) { text = "angry"; }
					else { text = "nature"; }
				}
				stringstream sstr;
				string temp="";
				sstr << res_xielv;
				sstr >> temp;
				sstr.flush();
				CString c(temp.c_str());
				OutputDebugString(c);
				OutputDebugString(_T("：斜率@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"));
				
				sstr << mouth_height;
				temp = "";
				sstr >> temp;
				sstr.flush();
				CString g(temp.c_str());
				OutputDebugString(g);
				OutputDebugString(_T("：mouth_height@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"));
				

				sstr << eye_hight;
				temp = "";
				sstr >> temp;
				sstr.flush();
				CString h(temp.c_str());
				OutputDebugString(h);
				OutputDebugString(_T("：eight_height@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"));

				CString kk(text.c_str());
				OutputDebugString(kk);
				sstr.flush();
				OutputDebugString(_T("：最终的表情判定@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"));
				
				OutputDebugString(_T("================================================================\n"));
				
				
				
				int font_face = cv::FONT_HERSHEY_COMPLEX;
				double font_scale = 1;
				int thickness = 2;
				int baseline;
				//获取文本框的长宽
				cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

				//将文本框居中绘制
				cv::Point origin;


				cv::rectangle(frame, cv::Rect(faces[j].left(), faces[j].top(), faces[j].width(), faces[j].width()), cv::Scalar(0, 0, 255), 1, 1, 0);//画矩形框
				origin.x = faces[j].left();
				origin.y = faces[j].top();
				cv::putText(frame, text, origin, font_face, font_scale, cv::Scalar(255, 0, 0), thickness, 2, 0);//给图片加文字
				
				

			}

		}
		//imshow("temp", frame);



	}
	catch (serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}

	*/

}
	else if (signal_point == 4) {
	Mat src_Image, gray_Image;
	src_Image = frame;
	//imshow("原图", src_Image);
	cvtColor(src_Image, gray_Image, CV_BGR2GRAY);
	const int MEDIAN_BLUR_FILTER_SIZE = 7;
	medianBlur(gray_Image, gray_Image, MEDIAN_BLUR_FILTER_SIZE);
	Mat edges_Image;
	const int LAPLACIAN_FILTER_SIZE = 5;
	Laplacian(gray_Image, edges_Image, CV_8U, LAPLACIAN_FILTER_SIZE);
	//imshow("均衡化前", edges_Image);
	Mat mask_Image;
	const int EDGES_THRESHOLD = 80;
	threshold(edges_Image, mask_Image, EDGES_THRESHOLD, 255, THRESH_BINARY_INV);
	//imshow("素描图", mask_Image);
	Size size = src_Image.size();
	Size smallSize;
	smallSize.width = size.width / 2;
	smallSize.height = size.height / 2;
	Mat small_Image = Mat(smallSize, CV_8UC3);
	cv::resize(src_Image, small_Image, smallSize, 0, 0, INTER_LINEAR);
	Mat tmp_Image = Mat(smallSize, CV_8UC3);
	int repetitions = 7;
	for (int i = 0; i < repetitions; i++)
	{
		int ksize = 9;
		double sigmaColor = 9;
		double sigmaSpace = 7;
		bilateralFilter(small_Image, tmp_Image, ksize, sigmaColor, sigmaSpace);
		bilateralFilter(tmp_Image, small_Image, ksize, sigmaColor, sigmaSpace);
	}
	Mat big_Image;
	cv::resize(small_Image, big_Image, size, 0, 0, INTER_LINEAR);
	Mat dst_Image = Mat(size, CV_8UC3);
	dst_Image.setTo(0);
	big_Image.copyTo(dst_Image, mask_Image);
	frame=dst_Image;
	
	
     }
	else if(signal_point==5){
	
	MultilPoolFunction( frame, 0);
	
}
	else if (signal_point == 6) {
	
	MultilPoolFunction( frame, 1);
	
}
	else if (signal_point == 7) {
	
	MultilPoolFunction(frame, 2);
	
}
	else if (signal_point == 8) {
	  
	   MultilPoolFunction(frame,3);
	   
}
	else {
		if (frame.channels() == 4)
			cvtColor(frame, frame, COLOR_BGRA2BGR);

		//! [Prepare blob]  
		Mat inputBlob = blobFromImage(frame, inScaleFactor,//https://blog.csdn.net/baidu_38505667/article/details/100168965?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase  预处理 减去均值
			Size(inWidth, inHeight), meanVal, false, false); //Convert Mat to batch of images  
															 //! [Prepare blob]  

		try
		{
			
			faceNet.setInput(inputBlob, "data"); //set the network input  
										 //! [Set input blob]  
		}
		catch (const std::exception&)
		{

		}													 //! [Set input blob]  
		

										 //! [Make forward pass]  
		Mat detection = faceNet.forward("detection_out"); //compute output  
													  //! [Make forward pass]  

		std::vector<double> layersTimings;
		double freq = getTickFrequency() / 1000;
		double time = faceNet.getPerfProfile(layersTimings) / freq;

		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		   ostringstream ss;
		//ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";
		//cv::putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));

		float confidenceThreshold = 0.5;

		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);

			if (confidence > confidenceThreshold)
			{
				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

				Rect object((int)xLeftBottom, (int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));
				//检测年龄 性别
				string age;
				string sex;
				try
				{
					Mat face = frame(object);
					//imshow("face",face);

					 age= predict_age(ageNet, face);

					 sex = predict_gender(genderNet, face);

					//endl
					cv::rectangle(frame, object, Scalar(0, 255, 0));
				}
				catch (const std::exception&e)
				{
					string error = e.what();
					CString temp ( error.c_str());

					OutputDebugString(temp);
				}
				

				//ss.str("");
				ss << confidence;
				String conf(ss.str());
				String label = "Face: " + conf;
				xingbie = sex;
				int baseLine = 0;
				Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				cv::rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
					Size(labelSize.width, labelSize.height + baseLine)),
					Scalar(255, 255, 255), CV_FILLED);
				cv::putText(frame, label, Point(xLeftBottom, yLeftBottom),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			}
		}
		//cv::imshow("detections", frame);
		//resize(frame, frame, Size(m_rect.Width(), m_rect.Height()));
		//imshow("temp", frame);




	}



	if (signal_point==5||signal_point==3|| signal_point==6|| signal_point==7|| signal_point==8){

		CRect rect;
		GetDlgItem(IDC_STATIC)->GetClientRect(&rect);
		Rect dst(rect.left, rect.top, rect.right, rect.bottom);
		Mat result;
		cv::resize(poolMat, result, cv::Size(rect.Width(), rect.Height()));

		imshow("view", result);

		signal_point = 0;
	}

	Mat temp;
	cv::flip(frame, temp, 1);//水平翻转


	IplImage img = IplImage(temp);
	drawPicToHDC(&img);
	
	//CDialogEx::OnTimer(nIDEvent);//这是，把当前函数没有被处理的消息id，用默认的处理函数来处理。类似与 switch里的最后哪个defaul
}


void CFaceRecognitionDlg::OnBnClickedCancel()
{
	// TODO: 在此添加控件通知处理程序代码
	KillTimer(1);            //这里关闭摄像头的方式  ，只是将时间函数停掉
	//cap.release();

	//CDialogEx::OnCancel();  关闭窗口
}


void CFaceRecognitionDlg::OnBnClickedButton1()
{
	// TODO: 在此添加控件通知处理程序代码
	signal_point = 1;
	
}


void CFaceRecognitionDlg::OnBnClickedButton2()
{
	signal_point = 2;
	// TODO: 在此添加控件通知处理程序代码
}


void CFaceRecognitionDlg::OnBnClickedButton3()
{
	signal_point = 3;
	// TODO: 在此添加控件通知处理程序代码
}


void CFaceRecognitionDlg::OnBnClickedButton4()
{
	signal_point = 4;
	// TODO: 在此添加控件通知处理程序代码
}


void CFaceRecognitionDlg::OnBnClickedButton5()
{
	signal_point = 5;
	// TODO: 在此添加控件通知处理程序代码
}


void CFaceRecognitionDlg::OnBnClickedButton6()
{
	signal_point = 6;
	// TODO: 在此添加控件通知处理程序代码
}


void CFaceRecognitionDlg::OnBnClickedButton7()
{
	signal_point = 7;
	// TODO: 在此添加控件通知处理程序代码
}


void CFaceRecognitionDlg::OnBnClickedButton8()
{
	signal_point = 8;
	// TODO: 在此添加控件通知处理程序代码
}


HBRUSH CFaceRecognitionDlg::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor)
{
	HBRUSH hbr = CDialogEx::OnCtlColor(pDC, pWnd, nCtlColor);

	// TODO:  在此更改 DC 的任何特性
	CBitmap bitmap;
	bitmap.LoadBitmapW(IDB_BACK);
	m_brush.CreatePatternBrush(&bitmap);
	// TODO:  如果默认的不是所需画笔，则返回另一个画笔
	return m_brush;
}


void CFaceRecognitionDlg::OnStnClickedImageShow()
{
	// TODO: 在此添加控件通知处理程序代码
}


void CFaceRecognitionDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值

	  // TODO: 在此添加消息处理程序代码和/或调用默认值
	CSliderCtrl   *pSlidCtrl = (CSliderCtrl*)GetDlgItem(IDC_SLIDER1);
	//m_int 即为当前滑块的值。
	m_int = 0.1*pSlidCtrl->GetPos();//取得当前位置值  
	CDialogEx::OnHScroll(nSBCode, nPos, pScrollBar);
	
}


void CFaceRecognitionDlg::OnEnChangeEdit1()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}
