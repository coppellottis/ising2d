# Compilatore
CC = gcc

# Opzioni compilazione
CFLAGS = -Wall -Wextra -O2 -Iinclude

# -Wall : attiva tutti i warning
# -Wextra : aggiunge altri controlli
# -O2 : ottimizzazione del codice
# -Iinclude : cerca i file .h anche anche nella cartella /include/.


# Nome eseguibile
TARGET = ising


# File sorgenti
SRC = \
src/main.c \
src/lattice.c \
src/pcg_basic.c \
src/metropolis.c \
src/wolff.c \
src/observables.c \
src/simulation.c \
src/io.c


# Conversione .c -> .o
OBJ = $(SRC:.c=.o)

# Prende i file in SRC e li sostituisce da .c a .o (automatizza la cosa)


# Regola principale
$(TARGET): $(OBJ) # gcc nome_file.o
	$(CC) $(OBJ) -o $(TARGET)
# La seconda riga linka gli eseguibili 


# Compilazione dei singoli file
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@


# Pulizia
clean:
	rm -f $(OBJ) $(TARGET)