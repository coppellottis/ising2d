# Compilatore

CC = gcc

# Opzioni compilazione

CFLAGS = -Wall -Wextra -O2 -Iinclude -fopenmp

# -Wall : attiva tutti i warning
# -Wextra : aggiunge altri controlli
# -O2 : ottimizzazione del codice
# -Iinclude : cerca i file .h nella cartella /include/

# Nome eseguibile

TARGET = ising

# File sorgenti

SRC = src/main.c \
      src/lattice.c \
      src/pcg_basic.c \
      src/metropolis.c \
      src/wolff.c \
      src/observables.c \
      src/simulation.c \
      src/io.c

# Cartella degli oggetti

BUILDDIR = build

# Conversione src/*.c -> build/*.o

OBJ = $(patsubst src/%.c,$(BUILDDIR)/%.o,$(SRC))

# Regola principale: link degli oggetti nell'eseguibile

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) -lm -fopenmp
# -lm linka la libreria matematica <math.h>

# Compilazione dei singoli file .c -> .o

$(BUILDDIR)/%.o: src/%.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Pulizia

clean:
	rm -rf $(BUILDDIR) $(TARGET)