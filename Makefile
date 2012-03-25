CC=gcc
CFLAGS=-O3 -D_XOPEN_SOURCE -Wall -std=c99 -I. `pkg-config --cflags glib-2.0`
LIBS=`pkg-config --libs glib-2.0` -lm -lgsl -lgslcblas

wkmeans: wkmeans.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

wkmeans.o: wkmeans.c wkmeans.h
	$(CC) -c $(CFLAGS) $< -o $@
