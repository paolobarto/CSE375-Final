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
  for(int i=0; i<8; i++)
  {
    cout<<"msg: "<<msg[i]<<endl;
  }

  msg.resize(LEN_DATABLOCK+8, '\0');

  for (int i = 0; i < 8; i++)
  {
    cout << "msg: " << msg[i] << endl;
  }

  bool success = send(sd, msg.data(), LEN_DATABLOCK+8, 0);
  if(!success)
  {
    cout << "Error sending message" << endl;
    return;
  }

  vector<uint8_t> response(LEN_DATABLOCK+8);
  int BUFFER_SIZE = 1024;
  uint8_t buffer[BUFFER_SIZE];

  int bytesRead;
  while ((bytesRead = recv(sd, buffer, BUFFER_SIZE, 0)) > 0)
  {
    for (int i = 0; i < bytesRead; ++i)
    {
      response[bytesRead++] = buffer[i];
      // cout << (uint8_t)buffer[i];
    }
    if (bytesRead >= 8 + LEN_DATABLOCK)
      break;
  }
  network->UpdateWeights(response);
}

void req_update(int sd, NeuralNetwork *network)
{
  vector<uint8_t> msg(LEN_DATABLOCK+8);
  string cmd = REQ_UPDT;
  msg.insert(msg.begin(), cmd.begin(), cmd.end());

  msg = network->ExtractWeights(REQ_UPDT);

  bool success = send(sd, (char*)&msg, LEN_DATABLOCK+8, 0);
  if(!success)
  {
    cout << "Error sending message" << endl;
    return;
  }
}

// /// Return a message ecrypted with an RSA PUB key
// ///
// /// @param pubkey The public key to encrypt with
// /// @param msg    The message to encrypt
// ///
// /// @return A vector with the encrypted message, or an empty vector on error
// std::vector<uint8_t> rsa_encrypt(EVP_PKEY *pubkey, const std::vector<uint8_t> &msg);

// /// req_key() writes a request for the server's key on a socket descriptor.
// /// When it gets a key back, it writes it to a file.
// ///
// /// @param sd      The open socket descriptor for communicating with the server
// /// @param keyfile The name of the file to which the key should be written
// void req_key(int sd, const string &keyfile)
// {
//   // need to pad the req_key with \0s
//   vector<uint8_t> key_block(REQ_KEY.begin(), REQ_KEY.end());
//   key_block.resize(LEN_RKBLOCK, '\0');

//   bool success = send_reliably(sd, key_block);
//   if (!success)
//   {
//     cout << "Error sending key request" << endl;
//     return;
//   }

//   vector<uint8_t> key_response(LEN_RSA_PUBKEY);
//   success = reliable_get_to_eof_or_n(sd, key_response.begin(), LEN_RSA_PUBKEY);

//   if (!success)
//   {
//     cout << "Error receiving key response" << endl;
//     return;
//   }

//   success = write_file(keyfile, key_response, 0);
//   if (!success)
//   {
//     cout << "Error writing key to file" << endl;
//     return;
//   }
// }

// /// req_reg() sends the REG command to register a new user
// ///
// /// @param sd      The open socket descriptor for communicating with the server
// /// @param pubkey  The public key of the server
// /// @param user    The name of the user doing the request
// /// @param pass    The password of the user doing the request
// void req_reg(int sd, EVP_PKEY *pubkey, const string &user, const string &pass,
//              const string &, const string &)
// {
//   vector<string> args{user, pass};
//   vector<uint8_t> response = send_cmd(sd, pubkey, REQ_REG, generate_ablock(args));
// }

// /// req_bye() writes a request for the server to exit.
// ///
// /// @param sd      The open socket descriptor for communicating with the server
// /// @param pubkey  The public key of the server
// /// @param user    The name of the user doing the request
// /// @param pass    The password of the user doing the request
// void req_bye(int sd, EVP_PKEY *pubkey, const string &user, const string &pass, const string &, const string &)
// {
//   vector<string> args{user, pass};
//   vector<uint8_t> response = send_cmd(sd, pubkey, REQ_BYE, generate_ablock(args));
// }

// /// req_sav() writes a request for the server to save its contents
// ///
// /// @param sd      The open socket descriptor for communicating with the server
// /// @param pubkey  The public key of the server
// /// @param user    The name of the user doing the request
// /// @param pass    The password of the user doing the request
// void req_sav(int sd, EVP_PKEY *pubkey, const string &user, const string &pass,
//              const string &, const string &)
// {
//   vector<string> args{user, pass};
//   vector<uint8_t> response = send_cmd(sd, pubkey, REQ_SAV, generate_ablock(args));
// }

// /// req_cpw() sends the CPW command to change the password of a user
// ///
// /// @param sd      The open socket descriptor for communicating with the server
// /// @param pubkey  The public key of the server
// /// @param user    The name of the user doing the request
// /// @param pass    The password of the user doing the request
// /// @param newpass The new password
// void req_cpw(int sd, EVP_PKEY *pubkey, const string &user, const string &pass,
//              const string &newpass, const string &)
// {
//   vector<string> args{user, pass, newpass};
//   vector<uint8_t> response = send_cmd(sd, pubkey, REQ_CPW, generate_ablock(args));
// }

// /// req_set() sends the SET command to set the content for a user
// ///
// /// @param sd      The open socket descriptor for communicating with the server
// /// @param pubkey  The public key of the server
// /// @param user    The name of the user doing the request
// /// @param pass    The password of the user doing the request
// /// @param setfile The file whose contents should be sent
// void req_set(int sd, EVP_PKEY *pubkey, const string &user, const string &pass,
//              const string &setfile, const string &)
// {
//   vector<uint8_t> file_contents_vec = load_entire_file(setfile);
//   string file_contents(file_contents_vec.begin(), file_contents_vec.end());
//   vector<string> args{user, pass, file_contents};
//   vector<uint8_t> response = send_cmd(sd, pubkey, REQ_SET, generate_ablock(args));
// }

// /// req_get() requests the content associated with a user, and saves it to a
// /// file called <user>.file.dat.
// ///
// /// @param sd      The open socket descriptor for communicating with the server
// /// @param pubkey  The public key of the server
// /// @param user    The name of the user doing the request
// /// @param pass    The password of the user doing the request
// /// @param getname The name of the user whose content should be fetched
// void req_get(int sd, EVP_PKEY *pubkey, const string &user, const string &pass,
//              const string &getname, const string &)
// {
//   vector<string> args{user, pass, getname};
//   vector<uint8_t> response = send_cmd(sd, pubkey, REQ_GET, generate_ablock(args));
//   if(response.size()==0) {
//     return;
//   }
//   vector<uint8_t> file_length = vector<uint8_t>(response.begin(), response.begin() + 8);
//   uint64_t length;
//   memcpy(&length, file_length.data(), 8);
//   string file_name = getname + ".file.dat";
//   vector<uint8_t> file_response = vector<uint8_t>(response.begin() + 8, response.end());
//   string file_contents(response.begin() + 8, response.end());
//   bool success = write_file(file_name, file_response, 0);
//   if (!success)
//   {
//     cout << "Error writing all to file" << endl;
//     return;
//   }
// }

// /// req_all() sends the ALL command to get a listing of all users, formatted
// /// as text with one entry per line.
// ///
// /// @param sd      The open socket descriptor for communicating with the server
// /// @param pubkey  The public key of the server
// /// @param user    The name of the user doing the request
// /// @param pass    The password of the user doing the request
// /// @param allfile The file where the result should go
// void req_all(int sd, EVP_PKEY *pubkey, const string &user, const string &pass,
//              const string &allfile, const string &)
// {
//   vector<string> args{user, pass};
//   vector<uint8_t> response = send_cmd(sd, pubkey, REQ_ALL, generate_ablock(args));
//   vector<uint8_t> file_length = vector<uint8_t>(response.begin(), response.begin() + 8);
//   uint64_t length;
//   memcpy(&length, file_length.data(), 8);
//   vector<uint8_t> file_response = vector<uint8_t>(response.begin() + 8, response.end());
//   bool success = write_file(allfile, file_response, 0);
//   if (!success)
//   {
//     cout << "Error writing all to file" << endl;
//     return;
//   }
// }

// /// Send a message to the server, using the common format for secure messages,
// /// then take the response from the server, decrypt it, and return it.
// ///
// /// Many of the messages in our server have a common form (@rblock.@ablock):
// ///   - @rblock enc(pubkey, padR("CMD".aeskey.length(@msg)))
// ///   - @ablock enc(aeskey, @msg)
// ///
// /// @param sd  An open socket
// /// @param pub The server's public key, for encrypting the aes key
// /// @param cmd The command that is being sent
// /// @param msg The contents of the @ablock
// ///
// /// @returns a vector with the (decrypted) result, or an empty vector on error
// vector<uint8_t> send_cmd(int sd, EVP_PKEY *pub, const string &cmd, const vector<uint8_t> &msg)
// {

//   // cmd.
//   vector<uint8_t> rblock;
//   rblock.insert(rblock.end(), cmd.begin(), cmd.end());

//   vector<uint8_t> aes_key = create_aes_key();
//   if (aes_key.empty() || aes_key.size() != AES_KEYSIZE + AES_IVSIZE)
//   {
//     cout << "Error creating aes key" << endl;
//     return vector<uint8_t>();
//   }

//   // cmd.aeskey.
//   rblock.insert(rblock.end(), aes_key.begin(), aes_key.end());

//   // encrypt ablock, used for length
//   //  enc(aeskey, @msg)
//   EVP_CIPHER_CTX *CTX = create_aes_context(aes_key, true);
//   vector<uint8_t> ablock = aes_crypt_msg(CTX, msg);
//   EVP_CIPHER_CTX_free(CTX);

//   // cmd.aeskey.length(@msg)
//   vector<uint8_t> ablock_length(8);
//   uint64_t length = ablock.size();
//   memcpy(ablock_length.data(), &length, 8); // copy length of argument into a vector.
//   rblock.insert(rblock.end(), ablock_length.begin(), ablock_length.end());

//   // pad rblock.
//   bool successful = padr(rblock, LEN_RBLOCK_CONTENT);
//   if (!successful)
//   {
//     cout << "There was an error in padr()" << endl;
//   }

//   // send encrypted rblock
//   // enc(pubkey, padR("CMD".aeskey.length(@msg)))
//   vector<uint8_t> encrypted_rblock = rsa_encrypt(pub, rblock);
//   if (encrypted_rblock.empty())
//   {
//     cout << "Error encrypting rblock" << endl;
//     return vector<uint8_t>();
//   }
//   bool success = send_reliably(sd, encrypted_rblock);
//   if (!success)
//   {
//     cout << "Error sending encrypted rblock" << endl;
//     return vector<uint8_t>();
//   }

//   // send encrypted ablock.
//   // enc(aeskey, @msg)
//   success = send_reliably(sd, ablock);
//   if (!success)
//   {
//     cout << "Error sending encrypted ablock" << endl;
//     return vector<uint8_t>();
//   }

//   vector<uint8_t> response = reliable_get_to_eof(sd);

//   // create new context and decrypt response
//   EVP_CIPHER_CTX *CTXd = create_aes_context(aes_key, false);
//   vector<uint8_t> dcryptResponse = aes_crypt_msg(CTXd, response);
//   string response_str = string(dcryptResponse.begin(), dcryptResponse.end());
//   if (response_str == RES_OK)
//   {
//     cout << RES_OK << endl;
//   }
//   else if (response_str != RES_OK && response_str.find(RES_OK) != string::npos)
//   {
//     cout << RES_OK << endl;
//     response = vector<uint8_t>(dcryptResponse.begin() + 8, dcryptResponse.end());
//   }
//   else
//   {
//     cout << response_str << endl;
//     vector<uint8_t> empty;
//     return empty;
//   }
//   return response;
// }

// /// generate_ablock generates a ablock based on the parameters passed in
// /// @param args vector of strings containing the arguments to be passed into a block
// /// @return
// vector<uint8_t> generate_ablock(vector<string> &args)
// {
//   vector<uint8_t> ablock;
//   for (auto &arg : args)
//   {
//     vector<uint8_t> arg_length(8);
//     uint64_t length = arg.length();
//     memcpy(arg_length.data(), &length, 8); // copy lenght of argument into a vector.
//     ablock.insert(ablock.end(), arg_length.begin(), arg_length.end());
//     ablock.insert(ablock.end(), arg.begin(), arg.end());
//   }

//   // unencrypted ablock
//   return ablock;
// }

// /// fill blocks with random values to fit length of param
// /// @param vec vector to be filled with random values
// /// @param length the length to with to fill the vector
// /// @return boolean to indicate success of function
// bool padr(vector<uint8_t> &vec, int length)
// {
//   int missing_length = length - vec.size(); // missing length = (256) - ()
//   vector<uint8_t> rand_values(missing_length);
//   RAND_bytes(rand_values.data(), missing_length);
//   vec.insert(vec.end(), rand_values.begin(), rand_values.end());
//   return true;
// }

// /// Return a message ecrypted with an RSA PUB key
// ///
// /// @param pubkey The public key to encrypt with
// /// @param msg    The message to encrypt
// ///
// /// @return A vector with the encrypted message, or an empty vector on error
// vector<uint8_t> rsa_encrypt(EVP_PKEY *pubkey, const vector<uint8_t> &msg)
// {
//   vector<unsigned char> newMsg(msg.begin(), msg.end());

//   EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new(pubkey, NULL);

//   if (ctx == nullptr)
//   {
//     cout << "Error calling EVP_PKEY_CTX_new()" << endl;
//     return vector<uint8_t>();
//   }
//   if (1 != EVP_PKEY_encrypt_init(ctx))
//   {
//     EVP_PKEY_CTX_free(ctx);
//     cout << "Error calling EVP_PKEY_encrypt_init()" << endl;
//     return vector<uint8_t>();
//   }

//   size_t enc_count = 0;
//   if (1 != EVP_PKEY_encrypt(ctx, nullptr, &enc_count, msg.data(), msg.size()))
//   {
//     EVP_PKEY_CTX_free(ctx);
//     cout << "Error computing encrypted buffer size" << endl;
//     return vector<uint8_t>();
//   }

//   vector<uint8_t> enc(enc_count);
//   if (1 != EVP_PKEY_encrypt(ctx, enc.data(), &enc_count, msg.data(), msg.size()))
//   {
//     EVP_PKEY_CTX_free(ctx);
//     cout << "Error calling EVP_PKEY_encrypt()" << endl;
//     return vector<uint8_t>();
//   }
//   EVP_PKEY_CTX_free(ctx);

//   return enc;
// }