CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		RabbitCL.o

LIBS =

TARGET =	RabbitCL

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
