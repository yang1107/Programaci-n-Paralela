
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

//using namespace std;

unsigned int NCIUDADES;

int siguiente, anterior;
int rank, size;

MPI_Comm comunicadorCarga;	// Para la distribuci�n de la carga
MPI_Comm comunicadorCota;	// Para la difusi�n de una nueva cota superior detectada

main (int argc, char **argv) {
        MPI::Init(argc,argv);
	switch (argc) {
		case 3:		NCIUDADES = atoi(argv[1]);
					break;
		default:	std::cerr << "La sintaxis es: bbseq <tama�o> <archivo>" << std::endl;
					exit(1);
					break;
	}

	int** tsp0 = reservarMatrizCuadrada(NCIUDADES);
	tNodo	nodo,         // nodo a explorar
			lnodo,        // hijo izquierdo
			rnodo,        // hijo derecho
			solucion;     // mejor solucion
	bool fin,        // condicion de fin
		nueva_U;       // hay nuevo valor de c.s.
	int  U;             // valor de c.s.
	int iteraciones = 0;
	tPila pila;         // pila de nodos a explorar

	U = INFINITO;                  // inicializa cota superior
	InicNodo (&nodo);              // inicializa estructura nodo

  int tam = NCIUDADES * NCIUDADES;

  MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comunicadorCarga);
	MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comunicadorCota);


  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  siguiente = (rank+1)%size;
  anterior = (rank-1+size)%size;


  if(rank == 0){
    LeerMatriz (argv[2], tsp0);    // lee matriz de fichero

    fin = Inconsistente(tsp0);
  }

  // Compartimos la matriz entre todas los procesos
  MPI_Bcast(*tsp0, // Dato a compartir
          tam, // Numero de elementos que se van a enviar y recibir
          MPI_INT, // Tipo de dato que se compartira
          0, // Proceso raiz que envia los datos
          MPI_COMM_WORLD); // Comunicador utilizado (En este caso, el global)

/*
  std::cout << "Proceso con id " << rank << " recibido la matriz" << std::endl;

  printf ("-------------------------------------------------------------\n");
    for (int i=0; i<NCIUDADES; i++) {
      for (int j=0; j<NCIUDADES; j++) {
        printf ("%3d", tsp0[i][j]);
      }
      printf ("\n");
    }
  printf ("-------------------------------------------------------------\n");
*/


  if(rank != 0){
    fin = Inconsistente(tsp0);
    Equilibrado_Carga(&pila, &fin);
    if(!fin) pila.pop(nodo);
  }

/*
  if(!fin){
    printf ("no fin\n");
  }
  else{
    printf ("fin");
  }
*/
  while (!fin) {       // ciclo del Branch&Bound
    Ramifica (&nodo, &lnodo, &rnodo, tsp0);
    nueva_U = false;
    if (Solucion(&rnodo)) {
      if (rnodo.ci() < U) {    // se ha encontrado una solucion mejor
        U = rnodo.ci();
        nueva_U = true;
        CopiaNodo (&rnodo, &solucion);
      }
    }
    else {                    //  no es un nodo solucion
      if (rnodo.ci() < U) {     //  cota inferior menor que cota superior
        if (!pila.push(rnodo)) {
          printf ("Error: pila agotada\n");
          liberarMatriz(tsp0);
          exit (1);
        }
      }
    }
    if (Solucion(&lnodo)) {
      if (lnodo.ci() < U) {    // se ha encontrado una solucion mejor
        U = lnodo.ci();
        nueva_U = true;
        CopiaNodo (&lnodo,&solucion);
      }
    }
    else {                     // no es nodo solucion
      if (lnodo.ci() < U) {      // cota inferior menor que cota superior
        if (!pila.push(lnodo)) {
          printf ("Error: pila agotada\n");
          liberarMatriz(tsp0);
          exit (1);
        }
      }
    }
    if (nueva_U) pila.acotar(U);
    Equilibrado_Carga(&pila, &fin);
    if(!fin) pila.pop(nodo);
    iteraciones++;
  }


  std::cout << "Proceso " << rank << " iteraciones: " << iteraciones << std::endl;


  liberarMatriz(tsp0);

  MPI_Finalize();
}
