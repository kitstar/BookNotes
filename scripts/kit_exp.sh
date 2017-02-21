#!/bin/bash

### Constantis
username="kit"
password="19870817"

host_ip="192.168.1.120"
machine_list="machine_list.txt"
script_path="/home/kit/exp/dnn"
run_path="/home/kit/run/dnn"
python_path="~/anaconda2"

ps_port=8700
worker_port=8800
process_name='python'


### Utils
function remote_cmd()
{
    ssh -n -l ${username} ${1} "${2}"
}


function remote_bg_cmd()
{
    ssh -n -f -l ${username} ${1} "${2}" >/dev/null 2&>1
}


function count_machine()
{
    local total=0
    while IFS= read -r line; do
        if [ "${line:0:1}" != "#" ]
        then
            (( total += 1 ))
        fi
    done < ${machine_list}
    echo ${total}
}
 


### UDFs
install_python()
{
    echo "TODO"
}

gen_script()
{
    local ps_list=""
    local worker_list=""
    
    local index=0    
    while IFS= read -r line; do
        if [ "${line:0:1}" != "#" ]
        then
            if [ ${index} -gt 0 ]; then
                ps_list+=','
                worker_list+=','
            fi
            ps_list+="${line}:${ps_port}"
            worker_list+="${line}:${worker_port}"
            (( index += 1 ))
        fi
    done < ${machine_list}
    
    index=0
    while IFS= read -r line; do
        if [ "${line:0:1}" != "#" ]
        then
            remote_cmd ${line} "echo '#!/usr/bin/env bash' > ${run_path}/s.sh; chmod +x ${run_path}/s.sh; echo cd ${run_path} >> ${run_path}/s.sh; echo python cnn/multi_node_async_benchmark.py --ps_hosts=${ps_list} --worker_hosts=${worker_list} --job_name=ps --task_index=${index} >> ${run_path}/s.sh"
            
            remote_cmd ${line} "echo '#!/usr/bin/env bash' > ${run_path}/w.sh; chmod +x ${run_path}/w.sh; echo cd ${run_path} >> ${run_path}/w.sh; echo python cnn/multi_node_async_benchmark.py --ps_hosts=${ps_list} --worker_hosts=${worker_list} --job_name=worker --task_index=${index} >> ${run_path}/w.sh"
            (( index += 1 ))
        fi
    done < ${machine_list}
}


function start_server()
{
    while IFS= read -r line; do
        if [ "${line:0:1}" != "#" ]
        then
            echo "Start server process ${line}"
            remote_bg_cmd ${line} "${run_path}/s.sh &"
        fi
    done < ${machine_list}
    
}


function start_worker()
{
    local mc=$( count_machine )
    local index=0
    while IFS= read -r line; do
        if [ "${line:0:1}" != "#" ]
        then
            (( index += 1))
            echo "Start worker process ${line}"
            if [ ${index} -lt ${mc} ]
            then
                remote_bg_cmd ${line} "${run_path}/w.sh &"
            else
                remote_cmd ${line} "${run_path}/w.sh"
            fi
        fi
    done < ${machine_list}
    
}


function perf_server()
{
    echo nothing
}


function perf_worker()
{
    ps aux | grep "python" | grep "=worker" | awk '{print $2}' | xargs -i perf record -o "worker.perf" -p {}
}


### Functions
function kill_process()
{
    while IFS= read -r line; do
        if [ "${line:0:1}" != "#" ]
        then
            remote_cmd ${line} "pkill -9 -u ${username} -f ${process_name}"
        fi
    done < ${machine_list}
}


setup () 
{
    install_python
}


function auth()
{
    if [ ! -f "~/.ssh/id_rsa.pub" ]; then
        ssh-keygen -t rsa
    fi

    while IFS= read -r line; do
#        if [ "${line}" != "${host_ip}" ] && [ "${line:0:1}" != "#" ]
        if [ "${line:0:1}" != "#" ]
        then
            echo "copying authority to ${line}"
            ssh-copy-id -i ~/.ssh/id_rsa.pub "${line}"
        fi
    done < ${machine_list}
}


copy_script()
{
    while IFS= read -r line; do
        if [ "${line:0:1}" != "#" ]
        then
            echo "copying scripts to ${line}"
            remote_cmd ${line} "mkdir -p ${run_path}"
            scp -r -q -C ${script_path}/* ${username}@${line}:${run_path}
        fi
    done < ${machine_list}
}


startall()
{
    kill_process
    start_server
    start_worker
}


usage ()
{
    echo "Call Usage"
}



############################
# main ()                  #

case "${1}" in
    "auth") auth
    ;;

    "setup" | "st") setup
    ;;

    "genscript" | "gs") gen_script
    ;;

    "copyscript" | "cs") copy_script
    ;;

    "startall" | "sa") startall
    ;;

    "startserver" | "ss") start_server
    ;;

    "perfserver" | "ps") perf_server
    ;;

    "startworker" | "sw") start_worker
    ;;

    "perfworker" | "pw") perf_worker
    ;;

    "kill" | "k") kill_process
    ;;

    "help" | "h" | "") usage
    ;;

    *) echo "Invalid command ${1}!"
    ;;

esac
