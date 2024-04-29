#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <sys/socket.h>

#include "../common/contextmanager.h"
#include "../common/protocol.h"

#include "requests.h"


using namespace std;



void req_network(int sd, NeuralNetwork *network)
{
 
  vector<uint8_t> msg(REQ_TRAIN.begin(), REQ_TRAIN.end());

  msg.resize(LEN_DATABLOCK+8, '\0');
  cout<<"Message Requesting Train Size: "<<msg.size()<<endl;
  bool success = send(sd, msg.data(), LEN_DATABLOCK+8, 0);
  if(!success)
  {
    cout << "Error sending message" << endl;
    return;
  }
  cout<<"Message Sent"<<endl;

  vector<uint8_t> response(LEN_DATABLOCK + 8);
  cout<<"Size of response empty: "<<response.size()<<endl;

  // int BUFFER_SIZE = LEN_DATABLOCK + 8;
  // uint8_t buffer[BUFFER_SIZE];
  // int bytesRead=0;

  // while ((bytesRead = recv(sd, buffer, BUFFER_SIZE, 0)) > 0)
  // {
  //   cout<<"Bytes Read: "<<bytesRead<<endl;
  //   for (int i = 0; i < bytesRead; ++i)
  //   {
  //     //cout<<"Bytes Read: "<<bytesRead<<" buffer[i]: "<<(int)buffer[i]<<endl;
  //     response[i] = buffer[i];
  //     //cout << buffer[i];
  //   }
  //   if (bytesRead >= 8 + LEN_DATABLOCK)
  //     break;
  //}

  int remain = (int)response.size();
  unsigned char *next_byte = &*response.begin();
  int total = 0;
  while (remain)
  {
    int rcd = recv(sd, next_byte, remain, 0);
    if (rcd <= 0)
    {
      //cout<<"error Reading information"<<endl;
      continue;
    }
    else
    {
      next_byte += rcd;
      remain -= rcd;
      total += rcd;
    }
  }
  //return total;
  cout<<"Size of total: "<<total<<endl;

  //recv(sd, response.data(), LEN_DATABLOCK + 8, 0); 
  cout<<"Size of returned NN: "<<response.size()<<endl;
  cout<<"Updating Weights"<<endl;
  network->UpdateWeights(response);
}

void req_update(int sd, NeuralNetwork *network)
{
  vector<uint8_t> msg(LEN_DATABLOCK+8);
  string cmd = REQ_UPDT;
  msg.insert(msg.begin(), cmd.begin(), cmd.end());

  msg = network->ExtractWeights(REQ_UPDT);

  cout<<"Sending Updated Weights"<<endl;
  bool success = send(sd, msg.data(), LEN_DATABLOCK+8, 0);
  if(!success)
  {
    cout << "Error sending message" << endl;
    return;
  }
  cout<<"Message Sent"<<endl;
  vector<uint8_t> response(LEN_DATABLOCK + 8);

  int remain = (int)response.size();
  unsigned char *next_byte = &*response.begin();
  int total = 0;
  while (remain)
  {
    int rcd = recv(sd, next_byte, remain, 0);
    
    string cmd(response.begin(), response.begin() + 8);
    if(cmd != RESP_UPDATE)
    {
      
      continue;
    }

    if (rcd <= 0)
    {
      continue;
      //cout << "error Reading information" << endl;
    }
    else
    {
      next_byte += rcd;
      remain -= rcd;
      total += rcd;
    }
    cout << "Size of total: " << total << endl; 
    cout << "Size of returned NN: " << response.size() << endl;
    cout << "Updating Weights" << endl;
    network->UpdateWeights(response);
  }
}