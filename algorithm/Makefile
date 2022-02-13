OBJ = main.o data.o def.o wsd.o
CC = gcc
CXX = g++
CFLAGS = -Wall -O3 -g


all: $(OBJ)
	$(CXX) $(CFLAGS) -o wsd $(OBJ)

main.o: main.cpp data.h def.h wsd.h
	$(CXX) $(CFLAGS) -c main.cpp

def.o: def.cpp def.h
	$(CXX) $(CFLAGS) -c def.cpp

data.o: data.cpp data.h def.h
	$(CXX) $(CFLAGS) -c data.cpp

wsd.o: wsd.cpp wsd.h def.h
	$(CXX) $(CFLAGS) -c wsd.cpp


.PHONY: clean
clean:
	rm wsd $(OBJ)
