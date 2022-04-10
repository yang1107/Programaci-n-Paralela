#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

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
            *miFila, // La fila que almacena localmente un proceso
            *comprueba, // Guarda el resultado final (calculado secuencialmente), su valor
                        // debe ser igual al de 'y'
            *y_local;

    double tInicio, // Tiempo en el que comienza la ejecucion
            tFin; // Tiempo en el que acaba la ejecucion

    double t0, t1;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numeroProcesadores);
    MPI_Comm_rank(MPI_COMM_WORLD, &idProceso);

    n = atoi(argv[1]);

    A = new int *[n]; // Reservamos n filas
    x = new int [n];
    y_local = new int [n/numeroProcesadores];

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

    int nelem = (n*n)/numeroProcesadores;


    // Reservamos espacio para la fila local de cada proceso
    miFila = new int [nelem];

    // Repartimos una fila por cada proceso, es posible hacer la reparticion de esta
    // manera ya que la matriz esta creada como un unico vector.

    MPI_Scatter(A[0], // Matriz que vamos a compartir
            nelem, // Numero de columnas a compartir
            MPI_INT, // Tipo de dato a enviar
            miFila, // Vector en el que almacenar los datos
            nelem, // Numero de columnas a recibir
            MPI_INT, // Tipo de dato a recibir
            0, // Proceso raiz que envia los datos
            MPI_COMM_WORLD); // Comunicador utilizado (En este caso, el global)


    // Compartimos el vector entre todas los procesos
    MPI_Bcast(x, // Dato a compartir
            n, // Numero de elementos que se van a enviar y recibir
            MPI_INT, // Tipo de dato que se compartira
            0, // Proceso raiz que envia los datos
            MPI_COMM_WORLD); // Comunicador utilizado (En este caso, el global)

    // Hacemos una barrera para asegurar que todas los procesos comiencen la ejecucion
    // a la vez, para tener mejor control del tiempo empleado
    MPI_Barrier(MPI_COMM_WORLD);
    // Inicio de medicion de tiempo
    tInicio = MPI_Wtime();

    int subFinal = 0;
    int pos=0;
    int cont=0;
    for (unsigned int i = 0; i < nelem; i++) {
        subFinal += miFila[i] * x[cont];
        cont++;
        if(cont>=n){
          cont=0;
          y_local[pos]=subFinal;
          pos++;
          subFinal=0;
        }

    }

    // Otra barrera para asegurar que todas ejecuten el siguiente trozo de c�digo lo
    // mas proximamente posible
    MPI_Barrier(MPI_COMM_WORLD);
    // fin de medicion de tiempo
    tFin = MPI_Wtime();

    // Recogemos los datos de la multiplicacion, por cada proceso sera un escalar
    // y se recoge en un vector, Gather se asegura de que la recolecci�n se haga
    // en el mismo orden en el que se hace el Scatter, con lo que cada escalar
    // acaba en su posicion correspondiente del vector.
    MPI_Gather(y_local, // Dato que envia cada proceso
            n/numeroProcesadores, // Numero de elementos que se envian
            MPI_INT, // Tipo del dato que se envia
            y, // Vector en el que se recolectan los datos
            n/numeroProcesadores, // Numero de datos que se esperan recibir por cada proceso
            MPI_INT, // Tipo del dato que se recibira
            0, // proceso que va a recibir los datos
            MPI_COMM_WORLD); // Canal de comunicacion (Comunicador Global)

    // Terminamos la ejecucion de los procesos, despues de esto solo existira
    // el proceso 0
    // Ojo! Esto no significa que los demas procesos no ejecuten el resto
    // de codigo despues de "Finalize", es conveniente asegurarnos con una
    // condicion si vamos a ejecutar mas codigo (Por ejemplo, con "if(rank==0)".
    MPI_Finalize();


    if (idProceso == 0) {

        unsigned int errores = 0;

        cout << "El resultado obtenido y el esperado son:" << endl;
        for (unsigned int i = 0; i < n; i++) {
            cout << "\t" << y[i] << "\t|\t" << comprueba[i] << endl;
            if (comprueba[i] != y[i])
                errores++;
        }

        delete [] y;
        delete [] comprueba;
        delete [] A[0];

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

    delete [] x;
    delete [] A;
    delete [] miFila;
    delete [] y_local;

}
