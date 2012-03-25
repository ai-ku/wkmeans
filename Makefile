CC=gcc
CFLAGS=-O3 -D_GNU_SOURCE -Wall -std=gnu99
LIBS=-lm

wkmeans: wkmeans.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

wkmeans.o: wkmeans.c wkmeans.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm wkmeans.o wkmeans
