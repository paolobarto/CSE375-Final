#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <queue>
#include <thread>
#include <unistd.h>
#include <mutex>
#include <shared_mutex>

#include "pool.h"
//#include "thread.h"


using namespace std;

class my_pool : public thread_pool {
public:

  vector<thread> threads;
  queue<int> connections; //queue of socket Descriptors
  condition_variable con;
  mutex lock;
  function<void()> end_func;
  atomic<bool> shutdown = false;
  function<bool(int, thread_pool*)> storedHandler;

  void setup(){
    int sd = -1;
    bool tempsh = false;
    while(!shutdown.load()){
      unique_lock<std::mutex> lk(this->lock); // locks the mutex
      cout<<"waiting"<<endl;
      this->con.wait(lk, [&](){return shutdown.load() || !connections.empty();}); //waits for a condition variable to tell that there are items in queue, then locks
      if(!this->connections.empty()){
        sd = this->connections.front(); //gets the first socket in queue, and saves it to sd
        this->connections.pop(); //gets rid of first socket (we are servicing)
        lk.unlock();
      }
      if (sd != -1 && !shutdown.load()) {
        tempsh = this->storedHandler(sd, this); //runs handler on sd
      }

      if (tempsh || shutdown.load()) {
        shutdown.store(true);
        this->con.notify_all();
      }
      
      if(sd != -1){
        cout<<"Closing Socket"<<endl;
        close(sd);
      }
    }
    end_func();
    
  }

  /// construct a thread pool by providing a size and the function to run on
  /// each element that arrives in the queue
  ///
  /// @param size    The number of threads in the pool
  /// @param handler The code to run whenever something arrives in the pool
  my_pool(int size, function<bool(int, thread_pool*)> handler) {

    this->threads.resize(size);

    this->storedHandler = handler;

    this->modelUpdateThread = thread([&](){return UpdateHandler();});

    for(int i=0; i<size; i++){ //set up all threads in pool
      this->threads[i] = thread([&](){return setup();});
    }
  }

  /// destruct a thread pool
  virtual ~my_pool() = default;

  /// Allow a user of the pool to provide some code to run when the pool decides
  /// it needs to shut down.
  ///
  /// @param func The code that should be run when the pool shuts down
  virtual void set_shutdown_handler(function<void()> func) {
    //this->trainingTreads->fetch_sub(1);
    cout<<"Current Training Threads: "<<this->trainingTreads->load()<<endl;
    this->end_func = func;
    return;
  }

  /// Allow a user of the pool to see if the pool has been shut down
  virtual bool check_active() {
    return !this->shutdown.load();
  }

  /// Shutting down the pool can take some time.  await_shutdown() lets a user
  /// of the pool wait until the threads are all done servicing clients.
  virtual void await_shutdown() {
    //lk.unlock();
    this->shutdown.store(true);
    this->con.notify_all();
    cout<<"notified"<<endl;
    for(size_t i = 0; i < this->threads.size(); i++){
      if(threads[i].joinable()){
        this->threads[i].join(); //join all the threads together, waiting for them to all close..
      }
    }
    end_func();
    return;
  }

  /// When a new connection arrives at the server, it calls this to pass the
  /// connection to the pool for processing.
  ///
  /// @param sd The socket descriptor for the new connection
  virtual void service_connection(int sd) {
    unique_lock<mutex> lk(this->lock); //main thread locks this
    if(!shutdown.load()){
      this->connections.push(sd);
      this->con.notify_one();
    }
  }

  void UpdateHandler()
  {
    while(!shutdown.load())
    {
      if(this->trainingTreads->load()>0 && this->respondedThreads->load() == this->trainingTreads->load())
      {
        cout<<"Models Responded\n";
        //unique_lock<shared_mutex> lk(this->modelLock);
        this->modelLock.lock();
        //update the model
        //this->sumNetwork->UpdateModel(this.trainingTreads->load());
        this->network->AverageWeights(this->sumNetwork, this->trainingTreads->load());
        this->network->PrintLayerAverage();

        this->test_func();

        //reset the respondedThreads
        //wait for the next update
        this->sumNetwork->ResetWeights();
        this->respondedThreads->store(0);
        this->trainingTreads->store(0);
        this->modelLock.unlock();
        this->updateCon.notify_all();
      }


    }
  }
};

/// Create a thread_pool object.
///
/// We use a factory pattern (with private constructor) to ensure that anyone...
/// not sure if i need to edit.
thread_pool *pool_factory(int size, function<bool(int, thread_pool*)> func, function<void()> testFunc, NeuralNetwork *network, NeuralNetwork *sumNetwork) {
  thread_pool *pool = new my_pool(size, func);
  pool->network = network;
  pool->sumNetwork = sumNetwork;
  pool->test_func = testFunc;
  return pool;
}
