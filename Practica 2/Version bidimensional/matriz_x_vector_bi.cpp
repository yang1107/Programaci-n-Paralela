#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <cmath>

using namespace std;

int main(int argc, char * argv[]) {

  if (argc==1)
   {
       cout << "Debes introducir el tamanio de la matriz (n)        --->" << endl;
       cout << "[360/720/1080/1440/1800]." << endl;
       return 1;
   }


    int numeroProcesadores,
            idProceso,
            n; // tamanio de la matriz a multiplicar

    int **A, // Matriz a multiplicar
            *x, // Vector que vamos a multiplicar
            *y, // Vector donde almacenamos el resultado
            *comprueba; // Guarda el resultado final (calculado secuencialmente), su valor
                        // debe ser igual al de 'y'

    double tInicio, // Tiempo en el que comienza la ejecucion
            tFin; // Tiempo en el que acaba la ejecucion
    double t0, t1;


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numeroProcesadores);
    MPI_Comm_rank(MPI_COMM_WORLD, &idProceso);

    n = atoi(argv[1]);

    A = new int *[n]; // Reservamos n filas
    x = new int [n];


/********************Distribucion de la matriz inicial entre los procesos******************************/

  MPI_Datatype MPI_BLOQUE;

  int raiz_P = sqrt(numeroProcesadores);
  int tam = n/raiz_P;


  //crear buffer de envio para almacenar los datos empaquetados
  int *buf_envio;
  buf_envio = new int [n*n];


    // Solo el proceso 0 ejecuta el siguiente bloque
    if (idProceso == 0) {

        A[0] = new int [n * n];
        for (unsigned int i = 1; i < n; i++) {
            A[i] = A[i - 1] + n;
        }
        // Reservamos especio para el resultado
        y = new int [n];

        // Rellenamos 'A' y 'x' con valores aleatorios
        srand(time(0));
      //  cout << "La matriz y el vector generados son " << endl;
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = 0; j < n; j++) {
              //  if (j == 0) cout << "[";
                A[i][j] = rand() % 1000;
              //  cout << A[i][j];
              //  if (j == n - 1) cout << "]";
              //  else cout << "  ";
            }
            x[i] = rand() % 100;
          //  cout << "\t  [" << x[i] << "]" << endl;
        }
      //  cout << "\n";

        //definir tipo de bloque cuadrado
        MPI_Type_vector(tam, tam, n, MPI_INT, &MPI_BLOQUE);
        //crear nuevo Tipo
        MPI_Type_commit(&MPI_BLOQUE);

        //empaqueta bloque a bloque en el buffer de envio
        int posicion,i, fila_P, columna_P,comienzo;
        for(i=0, posicion=0; i<numeroProcesadores;i++){
          //calcular la posicion de comienzo de cada submatriz
          fila_P=i/raiz_P;
          columna_P=i%raiz_P;
          comienzo=(columna_P*tam)+(fila_P*tam*tam*raiz_P);
          MPI_Pack(&A[fila_P*tam][columna_P*tam],1,MPI_BLOQUE,buf_envio,sizeof(int)*n*n, &posicion, MPI_COMM_WORLD);
        }

        //liberar tipo bloque
        MPI_Type_free(&MPI_BLOQUE);

        // Reservamos espacio para la comprobacion
        comprueba = new int [n];
        // Lo calculamos de forma secuencial

        t0 = clock();

        for (unsigned int i = 0; i < n; i++) {
            comprueba[i] = 0;
            for (unsigned int j = 0; j < n; j++) {
                comprueba[i] += A[i][j] * x[j];
            }
        }

        t1 = clock();
      } // Termina el trozo de codigo que ejecuta solo 0

      int *buf_recep;
      buf_recep = new int [tam*tam];

      //distribuir la matriz entre los procesos
      MPI_Scatter(buf_envio,sizeof(int)*tam*tam,MPI_PACKED,buf_recep,tam*tam,MPI_INT,0,MPI_COMM_WORLD);


  /***************Distribucion del vector x entre los procesos**********/
  //reservar espacio para vector x local
  int *x_local;
  x_local = new int [tam];

  //creacion del comunicador diagonal
  MPI_Comm comm_diagonal;
  int color;
  if(idProceso%(raiz_P+1)==0){
    color=1;
  }
  else{
    color=MPI_UNDEFINED;
  }

  MPI_Comm_split(MPI_COMM_WORLD // a partir del comunicador global.
            , color // los del mismo color entraran en el mismo comunicador
            , idProceso // // indica el orden de asignacion de rango dentro de los nuevos comunicadores
            , &comm_diagonal); // Referencia al nuevo comunicador creado.



  //distribuir n/raiz_P elementos del vector a los procesos del comunicador diagonal
  if(color==1){
    MPI_Scatter(x,tam,MPI_INT,x_local,tam,MPI_INT,0,comm_diagonal);
  }

  //creacion del comunicador columna
  MPI_Comm comm_columna;
  color=idProceso%raiz_P;

  MPI_Comm_split(MPI_COMM_WORLD // a partir del comunicador global.
            , color // los del mismo color entraran en el mismo comunicador
            , idProceso // // indica el orden de asignacion de rango dentro de los nuevos comunicadores
            , &comm_columna); // Referencia al nuevo comunicador creado.

  MPI_Bcast(x_local,
            tam,
            MPI_INT,
            color,
            comm_columna);

  /***************************Calculo de vector subvector_y local de cada proceso********************************/
  int *suby_local;
  suby_local = new int [tam];

  // Hacemos una barrera para asegurar que todas los procesos comiencen la ejecucion
  // a la vez, para tener mejor control del tiempo empleado
  MPI_Barrier(MPI_COMM_WORLD);
  // Inicio de medicion de tiempo
  tInicio = MPI_Wtime();

  int subFinal = 0;
  int pos=0;
  int cont=0;
  for (unsigned int i = 0; i < tam*tam; i++) {
      subFinal += buf_recep[i] * x_local[cont];
      cont++;
      if(cont>=tam){
        cont=0;
        suby_local[pos]=subFinal;
        pos++;
        subFinal=0;
      }
  }

  // Otra barrera para asegurar que todas ejecuten el siguiente trozo de c�digo lo
  // mas proximamente posible
  MPI_Barrier(MPI_COMM_WORLD);
  // fin de medicion de tiempo
  tFin = MPI_Wtime();


/**************************calculo del vector y local de cada fila *******************************/

int *y_local;
y_local = new int [tam];

//creacion del comunicador fila
MPI_Comm comm_fila;
color=idProceso/raiz_P;

MPI_Comm_split(MPI_COMM_WORLD // a partir del comunicador global.
          , color // los del mismo color entraran en el mismo comunicador
          , idProceso // // indica el orden de asignacion de rango dentro de los nuevos comunicadores
          , &comm_fila); // Referencia al nuevo comunicador creado.

int root_f=idProceso/raiz_P;

MPI_Reduce(suby_local, //buff que contiene elementos a enviar
           y_local, // buff para almacenar valores recibidos
           tam,  // tamanio de elementos a recibir ---> n/raiz_P
           MPI_INT,  // Tipo de dato
           MPI_SUM, // operacion
           root_f, // root
           comm_fila); //comunicador


/**************************recoger los resultados en el vector y del proceso 0 *******************/

if(idProceso%(raiz_P+1)==0){
  color=1;
}
else{
  color=0;
}

if(color==1){

MPI_Gather(y_local, // Dato que envia cada proceso
        tam, // Numero de elementos que se envian
        MPI_INT, // Tipo del dato que se envia
        y, // Vector en el que se recolectan los datos
        tam, // Numero de datos que se esperan recibir por cada proceso
        MPI_INT, // Tipo del dato que se recibira
        0, // proceso que va a recibir los datos
        comm_diagonal); // Canal de comunicacion
}

if(idProceso==0){

  unsigned int errores = 0;

  cout << "El resultado obtenido y el esperado son:" << endl;
  for (unsigned int i = 0; i < n; i++) {
      cout << "\t" << y[i] << "\t|\t" << comprueba[i] << endl;
      if (comprueba[i] != y[i])
          errores++;
  }

  delete [] y;
  delete [] A;
  delete [] comprueba;

  if (errores) {
      cout << "Hubo " << errores << " errores." << endl;
  } else {
    double tiempoSecuencial, tiempoParalelo, ganancia;
    tiempoSecuencial = (t1-t0)/CLOCKS_PER_SEC;
    tiempoParalelo = tFin - tInicio;
    ganancia = tiempoSecuencial/tiempoParalelo;
    cout << "No hubo errores" << endl;
    cout << "El tiempo tardado en algoritmo secuencial ha sido " << tiempoSecuencial << " segundos." << endl;
    cout << "El tiempo tardado en algoritmo paralelo   ha sido " << tiempoParalelo << " segundos." << endl;
    cout << "La ganancia que se ha obtenido es                 " << ganancia << " ." << endl;
  }
}

  MPI_Finalize();

  delete [] x;
  delete [] buf_envio;
  delete [] buf_recep;
  delete [] y_local;
  delete [] x_local;


}
