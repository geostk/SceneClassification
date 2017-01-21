all: CImg.h final.cpp BOW.h
	module load opencv; \
	g++ final.cpp  -o final -lX11 -lpthread `pkg-config opencv --cflags` `pkg-config opencv --libs` -I. -O3 -Isiftpp siftpp/sift.cpp
clean:
	rm final
