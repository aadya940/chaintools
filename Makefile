CC=gcc
CFLAGS=-Wall -Wextra -I.
LIBS=-lgsl -lgslcblas -lm

LIBRARY=libchaintools.a

SRCS=$(wildcard src/*.c)
OBJS=$(SRCS:.c=.o)

all: $(LIBRARY)

$(LIBRARY): $(OBJS)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(OBJS) $(LIBRARY)
