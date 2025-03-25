iiii```shell
function latest_file(){
  curl -L  "http://webhdfs-offline-lt.corp.kuaishou.com/webhdfs/v1/home/reco_5/mpi/products/mio_py_env?op=LISTSTATUS&user.name=mpi"  2>/dev/null | python3 -c "import json, sys;
files = json.load(sys.stdin)['FileStatuses']['FileStatus'];
for  f in files:
  print(f['pathSuffix']);
  " |sort | tail -n 1
}

latest=`latest_file`

curl  -L "http://webhdfs-offline-lt.corp.kuaishou.com/webhdfs/v1/home/reco_5/mpi/products/mio_py_env/$latest?op=OPEN&user.name=mpi" -o py_env.tgz

mkdir -p mio_py_env
pushd mio_py_env
tar zxvf ../py_env.tgz
/bin/rm -rf  ../py_env.tgz
popd
```
