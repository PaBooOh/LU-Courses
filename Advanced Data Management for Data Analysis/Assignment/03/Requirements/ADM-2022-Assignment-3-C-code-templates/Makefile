CC=gcc
CFLAGS=-Wformat=2 -Wformat-signedness -Wextra -Wall -Wpedantic -Werror -pedantic-errors -O3
BINARIES=\
	ADM-2022-A3-sort-branched-inplace \
	ADM-2022-A3-sort-predicated-inplace \
	ADM-2022-A3-sort-branched-outofplace \
	ADM-2022-A3-sort-predicated-outofplace

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

.PHONY: all clean

all: $(BINARIES)

clean:
	rm -v $(BINARIES)
