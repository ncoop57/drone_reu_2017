#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.h>

#include <sys/timeb.h>
using namespace cv;

int CLOCK()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,  &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

double _avgdur=0;
int _fpsstart=0;
double _avgfps=0;
double _fps1sec=0;

double avgdur(double newdur)
{
    _avgdur=0.98*_avgdur+0.02*newdur;
    return _avgdur;
}

double avgfps()
{
    if(CLOCK()-_fpsstart>1000)      
    {
        _fpsstart=CLOCK();
        _avgfps=0.7*_avgfps+0.3*_fps1sec;
        _fps1sec=0;
    }

    _fps1sec++;
    return _avgfps;
}

void process(Mat& frame)
{
    imshow("frame",frame);
}

int main(int argc, char** argv)
{
    int frameno=0;
    cv::Mat frame;
    cv::VideoCapture cap(0);
    for(;;)
    {
        cap>>frame;

        clock_t start=CLOCK();

        if(frame.data)process(frame);

        double dur = CLOCK()-start;
        printf("avg time per frame %f ms. fps %f. frameno = %d\n",avgdur(dur),avgfps(),frameno++ );
        if(waitKey(1)==27)
            exit(0);
    }
    return 0;
}
