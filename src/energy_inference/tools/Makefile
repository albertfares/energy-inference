CXX ?= g++
CXXFLAGS ?= -O2 -Wall -Wextra -std=c++17

all: sample_ina3221

sample_ina3221: ina3221_linux.o  sample_ina3221.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ 

ina3221_linux.o: ina3221_linux.cpp ina3221_linux.h
	$(CXX) $(CXXFLAGS) -c -o $@ ina3221_linux.cpp

demo.o: main.cpp ina3221_linux.h
	$(CXX) $(CXXFLAGS) -c -o $@ main.cpp

clean:
	rm -f *.o ina3221_demo
