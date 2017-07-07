#sudo apt-get install build-essential
#sudo apt-get install ffmpeg
#sudo apt-get install libav-tools
#sudo apt-get install libopencv-dev

CXX           = g++
CXXFLAGS      = -O2 -Wall -fopenmp -D__STDC_CONSTANT_MACROS `pkg-config --libs --cflags opencv`
LIBS          = -lm                     \
                -lpthread               \
                -lavutil                \
                -lavformat              \
                -lavcodec               \
                -lswscale
OBJS          = ardrone/ardrone.o \
                ardrone/command.o \
                ardrone/config.o  \
                ardrone/udp.o     \
                ardrone/tcp.o     \
                ardrone/navdata.o \
                ardrone/version.o \
                ardrone/video.o   \
		drone_optical_flow_gpu.o
PROGRAM       = test

$(PROGRAM):     $(OBJS)
		$(CXX) $(OBJS) -o $(PROGRAM) $(CXXFLAGS) $(LDFLAGS) $(LIBS) 

clean:;         rm -f *.o *~ $(PROGRAM) $(OBJS)

install:        $(PROGRAM)
		install -s $(PROGRAM) $(DEST)
