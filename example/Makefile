CXX=g++
CXXFLAGS=-std=c++11 -O3 -Wall -Wextra -pedantic
LDFLAGS=-larmadillo # -lmlpack
INCLUDES=-I../include
SRC=main.cc

BIN=example

.PHONY:example

example:
	${CXX} ${CXXFLAGS} ${SRC} ${LDFLAGS} -o ${BIN} ${INCLUDES}

clean:
	@rm -frv ${BIN}
