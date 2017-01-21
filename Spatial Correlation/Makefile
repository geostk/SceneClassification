all: ScaledImg.h final.cpp BOWClassifier.h
	module load opencv; \
	g++ final.cpp  -o final -lX11 -lpthread `pkg-config opencv --cflags` `pkg-config opencv --libs` -I. -O3 
clean:
	rm final
