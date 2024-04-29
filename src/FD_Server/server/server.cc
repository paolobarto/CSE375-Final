#include <iostream>
#include <libgen.h>
//#include <openssl/rsa.h>
#include <string>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fstream>

#include "../common/contextmanager.h"
//#include "../common/crypto.h"
//#include "../common/err.h"
//#include "../common/file.h"
//#include "../common/net.h"

#include "../common/pool.h"
#include "../common/NeuralNetwork.h"

#include "parsing.h"
//#include "storage.h"

//using namespace std;
volatile sig_atomic_t int_signal_received = 0;
int sd;

void int_handler(int) {
  cout<<"SIGINT received\n";
  close(sd);
  int_signal_received = 1;
}

/// arg_t represents the command-line arguments to the server
struct arg_t {
  int port;                    // The port on which to listen
  // string datafile;             // The file for storing all data
  // string keyfile;              // The file holding the AES key
  int threads = 1;             // Number of threads for the server to use
  // size_t num_buckets = 1024;   // Number of buckets for the server's hash tables
  // size_t quota_interval = 60;  // Seconds over which a quota is enforced
  // size_t quota_up = 1048576;   // K/V upload quota (bytes/interval)
  // size_t quota_down = 1048576; // K/V download quota (bytes/interval)
  // size_t quota_req = 16;       // K/V request quota (requests/interval)
  // size_t top_size = 4;         // Number of keys to track for TOP queries
  // string admin_name = "";      // Name of the administrator

  /// Construct an arg_t from the command-line arguments to the program
  ///
  /// @param argc The number of command-line arguments passed to the program
  /// @param argv The list of command-line arguments
  ///
  /// @throw An integer exception (1) if an invalid argument is given, or if
  ///        `-h` is passed in
  arg_t(int argc, char **argv) {
    long opt;
    while ((opt = getopt(argc, argv, "p:t:")) != -1) {
      switch (opt) {
      case 'p':
        port = atoi(optarg);
        break;
      case 't':
        threads = atoi(optarg);
        break;
      default: // on any error, print a help message.  This case subsumes `-h`
        throw 1;
        return;
      }
    }
  }

  /// Display a help message to explain how the command-line parameters for this
  /// program work
  ///
  /// @progname The name of the program
  static void usage(char *progname) {
    cout << basename(progname) << ": company user directory server\n"
         << "  -p [int]    Port on which to listen for incoming connections\n"
         << "  -t [int]    # of threads that server should use\n";
  }
};

int main(int argc, char **argv) {
  // Parse the command-line arguments
  //
  // NB: It would be better not to put the arg_t on the heap, but then we'd need
  //     an extra level of nesting for the body of the rest of this function.
  arg_t *args;
  try {
    args = new arg_t(argc, argv);
  } catch (int i) {
    arg_t::usage(argv[0]);
    return 1;
  }

  fstream testFile("../../data/MNIST_CSV/mnist_test.csv", ios::in);
  if (!testFile.is_open())
  {
    cout << "Error: File not found" << endl;
    return 1;
  }
   vector<MNIST_Image> testImages;

  string line;
  while (getline(testFile, line))
  {
    testImages.push_back(MNIST_Image(line));
  }
  testFile.close();

  // print the configuration
  cout << "Listening on port " << args->port << "\n";


  NeuralNetwork *network = new NeuralNetwork();
  network->CreateNetwork(784, 16, 10, 2);
  network->PrintLayerAverage();

  NeuralNetwork *sumNetwork = new NeuralNetwork();
  sumNetwork->CreateNetwork(784, 16, 10, 2);
  sumNetwork->ResetWeights();


  
  
  // Start listening for connections.
  //int sd = create_server_socket(args->port);

  sockaddr_in servAddr;
  bzero((char *)&servAddr, sizeof(servAddr));
  servAddr.sin_family = AF_INET;
  servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
  servAddr.sin_port = htons(args->port);

  sd = socket(AF_INET, SOCK_STREAM, 0);

  if (sd < 0) {
    perror("socket() failed");
    return 1;
  }

  int bindStatus = ::bind(sd, (sockaddr *)&servAddr, sizeof(servAddr));
  if( bindStatus < 0) 
  {
    perror("bind() failed");
    return 1;
  }


  ContextManager csd([&]() { close(sd); });
  // Create a thread pool that will invoke parse_request (from a pool thread)
  // each time a new socket is given to it.
  thread_pool *pool = pool_factory(args->threads, [&](int sd, thread_pool *poolParam) {
    return parse_request(sd, network, sumNetwork, poolParam);
  }, [&](){
    int correctAnswers = 0;
    for(size_t i=0;i<testImages.size();i++)
    {
      int guess = network->ForwardPropagateImage(testImages[i]);
      if(guess == testImages[i].label)
      {
        correctAnswers++;
      }
    }
    cout<<"Correct Answers: "<<correctAnswers<<endl;

  }, network, sumNetwork);

  pool->set_shutdown_handler([&]() {
    cout<<"Closing Socket Descriptor\n";
    pool->trainingTreads->fetch_sub(1);
    close(sd);
  });
// sockaddr_in clientAddr;
// socklen_t clientAddrSize = sizeof(clientAddr);
// int newSd = accept(sd, (struct sockaddr*)&clientAddr, &clientAddrSize);
// Start accepting connections and passing them to the pool.

signal(SIGINT, int_handler);

listen(sd, 5);
while (!int_signal_received) {
  sockaddr_in clientAddr;
  socklen_t clientAddrSize = sizeof(clientAddr);
  int newSd = accept(sd, (struct sockaddr*)&clientAddr, &clientAddrSize);

  if (newSd < 0) {
    perror("accept() failed");
    return 0;
    continue;
  }

  // Pass the new socket descriptor to the thread pool.
  pool->service_connection(newSd);
}


  // Start accepting connections and passing them to the pool.
  //accept_client(sd, *pool); //TODO accept new client

  // The program can't exit until all threads in the pool are done.
  cout<<"shutting down\n";
  pool->await_shutdown();
  //storage->shutdown();
  delete pool;
  delete args;
  delete network;
  cout << "Server terminated\n";
}
