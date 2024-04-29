#include <iostream>
#include <libgen.h>
#include <string>
#include <unistd.h>
#include <vector>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <fstream>
#include <random>
#include <algorithm>

#include "../common/contextmanager.h"
#include "../common/protocol.h"
#include "../common/NeuralNetwork.h"

#include "requests.h"

using namespace std;

void shuffleVector(std::vector<MNIST_Image> &vec)
{
  // Obtain a random seed from hardware
  std::random_device rd;

  // Seed the random number generator
  std::default_random_engine rng(rd());

  // Shuffle the vector
  std::shuffle(vec.begin(), vec.end(), rng);
}

int connect_to_server(int *&sd, sockaddr_in &sendSockAddr)
{
  cout<<"Sd before: "<<*sd<<endl;
  *sd = socket(AF_INET, SOCK_STREAM, 0);
  int status = connect(*sd,
                       (sockaddr *)&sendSockAddr, sizeof(sendSockAddr));
  if (status < 0)
  {
    cout << "Error connecting to socket!" << endl;
    return -1;
  }
  cout << "Connected to the server!" << endl;
  return 0;
}

/// These are all of the commands that the client supports, with descriptions
/// that are formatted nicely for printing in `usage()`.
static vector<pair<string, const char *>> commands = {
    {REQ_TRAIN, "             Return the current state of the network"},
    {REQ_UPDT, "             Send the newest updated weights"},};

/// arg_t represents the command-line arguments to the client
struct arg_t {
  string server = "";   // The IP or hostname of the server
  int port = 0;         // The port on which to connect to the server
  // string keyfile = "";  // The file for storing the server's public key
  // string username = ""; // The user's name
  // string userpass = ""; // The user's password
  // string command = "";  // The command to execute
  string arg1 = "";     // The first argument to the command (if any)
  string arg2 = "";     // The second argument to the command (if any)

  /// Construct an arg_t from the command-line arguments to the program
  ///
  /// @param argc The number of command-line arguments passed to the program
  /// @param argv The list of command-line arguments
  ///
  /// @throw An integer exception (1) if an invalid argument is given, or if
  ///        `-h` is passed in
  arg_t(int argc, char **argv) {
    // First, use getopt to parse the command-line arguments
    long opt;
    while ((opt = getopt(argc, argv, "s:p:")) != -1) {
      switch (opt) {
      case 'p': // port of server
        port = atoi(optarg);
        break;
      case 's': // hostname of server
        cout<<"server: "<<string(optarg)<<endl;
        server = string(optarg);
        break;
      case '1': // first argument
        arg1 = string(optarg);
        break;
      case '2': // second argument
        arg2 = string(optarg);
      default: // on any error, print a help message.  This case subsumes `-h`
        throw 1;
        return;
      }
    }

    // Validate that the argument to `-C` is valid, and accompanied by `-1` and
    // `-2` as appropriate
    // string args0[] = {REQ_BYE, REQ_SAV, REQ_REG};
    // string args1[] = {
    //     REQ_SET,
    //     REQ_GET,
    //     REQ_ALL,
    //     REQ_CPW,
    // };
    //string args2[] = {};
    // for (auto a : args0) {
    //   if (command == a) {
    //     if (arg1 != "" || arg2 != "")
    //       throw 1;
    //     return;
    //   }
    // }
    // for (auto a : args1) {
    //   if (command == a) {
    //     if (arg1 == "" || arg2 != "")
    //       throw 1;
    //     return;
    //   }
    // }
    // for (auto a : args2) {
    //   if (command == a) {
    //     if (arg1 == "" || arg2 == "")
    //       throw 1;
    //     return;
    //   }
    // }
   // throw 1;
  }

  /// Display a help message to explain how the command-line parameters for this
  /// program work
  ///
  /// @progname The name of the program
  static void usage(char *progname) {
    cout << basename(progname) << ": Federated Learning client\n\n";

    cout << " Required Configuration Parameters:\n";
    cout << "  -s [string] IP address or hostname of server\n"
         << "  -p [int]    Port to use to connect to server\n";
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
    cout<<"Error: "<<i<<endl;
    arg_t::usage(argv[0]);
    return 1;
  }

  // Connect to the server and perform the appropriate operation
  //int sd = connect_to_server(args->server, args->port);
  //cout << "Connecting to the server..." << endl;
  struct hostent *host = gethostbyname(args->server.c_str());
  sockaddr_in sendSockAddr;
  bzero((char *)&sendSockAddr, sizeof(sendSockAddr));
  sendSockAddr.sin_family = AF_INET;
  sendSockAddr.sin_addr.s_addr =
      inet_addr(inet_ntoa(*(struct in_addr *)*host->h_addr_list));
  sendSockAddr.sin_port = htons(args->port);
  int *clientSd = new int();
  //int *clientSd = new int(socket(AF_INET, SOCK_STREAM, 0));
  //int clientSd = socket(AF_INET, SOCK_STREAM, 0);

  // int status = connect(clientSd,
  //                      (sockaddr *)&sendSockAddr, sizeof(sendSockAddr));
  // if (status < 0)
  // {
  //   cout << "Error connecting to socket!" << endl;
  //   return -1;
  // }
  // cout << "Connected to the server!" << endl;

  ContextManager sdc([&]() { 
    //TODO look into if server can handle removing existance of training client
    close(*clientSd); 
    });

  // read csv
  // fstream testFile("../../../data/MNIST_CSV/mnist_test.csv", ios::in);
  // if (!testFile.is_open())
  // {
  //   cout << "Error: File not found" << endl;
  //   return 1;
  // }
  fstream trainFile("../../data/MNIST_CSV/mnist_train.csv", ios::in);
  if (!trainFile.is_open())
  {
    cout << "Error: Train File not found" << endl;
    return 1;
  }

  //vector<MNIST_Image> testImages;
  vector<MNIST_Image> trainImages;

  string line;
  // while (getline(testFile, line))
  // {
  //   testImages.push_back(MNIST_Image(line));
  // }
  while (getline(trainFile, line))
  {
    trainImages.push_back(MNIST_Image(line));
  }
  trainFile.close();
  //testFile.close();
  //cout << testImages.size() << endl;
  cout << trainImages.size() << endl;

  cout<<"Shuffling training data."<<endl;
  shuffleVector(trainImages);

  // train loop
  // first get the weights of the current network
  cout<<"Creating Neural Network."<<endl;
  NeuralNetwork *network = new NeuralNetwork();
  network->CreateNetwork(LEN_INPUTNODES, LEN_HIDDENNODES, LEN_OUTPUTNODES, NUM_HIDDENLAYERS);
  network->PrintLayerAverage();
  // then train the network
  const int EPOCHS = 10;
  const int DEBUGBATCH = 30000;

  for(int i=0; i<EPOCHS; i++)
  {
  cout << "-------- Epoch loop: " << i << endl;
  shuffleVector(trainImages);
  int status = connect_to_server(clientSd, sendSockAddr);
  if (status < 0)
  {
    cout << "Error connecting to server." << endl;
    return 1;
    }
    cout<<"Client Sd: "<<*clientSd<<endl;
    cout<<"Requesting network from server."<<endl;
    req_network(*clientSd, network);
    //return 1;
    cout<<"Network updated to: "<<endl;
    network->PrintLayerAverage();
    cout<<"Epoch "<<i+1<<" started."<<endl;
    for(size_t j=0; j<DEBUGBATCH; j++)
    {           
        int value = network->ForwardPropagateImage(trainImages[j]);
        if(value == -100)
        {
          cout<<"Error in Forward Propagation"<<endl;
          return 1;
        }
        if(value != trainImages[j].label) {

          network->BackPropagateImage(trainImages[j]);
        }

        if(j%10000==0){
          cout<<"Batch "<<j<<" finished."<<endl;
          network->PrintLayerAverage();
        }

        network->ResetValues();
    }
    cout<<"Epoch "<<i+1<<" finished."<<endl;
    cout<<"Sending updated weights to server."<<endl;
    status = connect_to_server(clientSd, sendSockAddr);
    if(status < 0)
    {
      cout<<"Error connecting to server."<<endl;
      return 1;
    }
    req_update(*clientSd, network);
  }


  // Figure out which command was requested, and run it
  // vector<string> cmd = {REQ_REG, REQ_BYE, REQ_SET, REQ_GET, REQ_ALL, REQ_SAV, REQ_CPW};
  // decltype(req_reg) *func[] = {req_reg, req_bye, req_set,
  //                              req_get, req_all, req_sav, req_cpw};
  // for (size_t i = 0; i < cmd.size(); ++i)
  //   if (args->command == cmd[i])
  //     func[i](sd, pubkey, args->username, args->userpass, args->arg1,
  //             args->arg2);
   delete args;
  return 0;
}
