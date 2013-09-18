CXX=g++
CXXFLAGS=-std=c++11 -Ofast -Wall -Wextra -pedantic
LDFLAGS=-larmadillo -ltbb
INCLUDES=-I../include
SRC=geneticMain.cc

BIN=example

.PHONY:example

example:
	${CXX} ${CXXFLAGS} ${SRC} ${LDFLAGS} -o ${BIN} ${INCLUDES}

clean:
	@rm -frv ${BIN}
