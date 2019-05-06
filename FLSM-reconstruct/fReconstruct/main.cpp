#define RAPIDJSON_SSE42

#include <queue>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <cstdlib>

#include "genvoxeltask.h"
#include "cellcountingtask.h"

using namespace cv;
using namespace std;
using namespace flsm;

int resolveInputFile(string fileName, TaskManager& taskmgr)
{
    try {
        ifstream jf(fileName);
        if(!jf.is_open()) throw 1;
        jf.ignore(std::numeric_limits<std::streamsize>::max());
        streamsize fl = jf.gcount();
        jf.clear();
        jf.seekg(0, jf.beg);
        char* buf = new char[fl + 1];
        jf.read(buf, fl);
        jf.close();
        buf[fl] = 0;
        rapidjson::Document doc;
        if(doc.Parse(buf).HasParseError()) {
            cerr << "Error " << doc.GetErrorOffset() << ":" << rapidjson::GetParseError_En(doc.GetParseError()) << endl;
            throw 2;
        }
        if(!doc.IsObject()) throw 2;
        if(!doc.HasMember("brains")) throw 2;
        if(!doc.HasMember("tasks")) throw 2;
        for(int i = 0; doc["tasks"].HasMember(to_string(i).c_str()); i++) {
            rapidjson::Value& tsk = doc["tasks"][to_string(i).c_str()];
            if(!tsk.HasMember("type")) continue;
            if(!tsk["type"].IsString()) continue;
            if(!tsk.HasMember("parameters")) continue;
            if(!tsk["parameters"].IsObject()) continue;
            Task* t;
            string type = tsk["type"].GetString();
            if(!type.compare("genVoxelTask")) t = new GenVoxelTask();
            else if(!type.compare("nullTask")) t = new nullTask();
            else if(!type.compare("cellCountingTask")) t = new CellCountingTask();
            else continue;
            rapidjson::Value& par = tsk["parameters"];
            t->setParameters(par);
            taskmgr.tasks.push_back(t);
        }
        for(int i = 0; doc["brains"].HasMember(to_string(i).c_str()); i++) {
            rapidjson::Value& brain = doc["brains"][to_string(i).c_str()];
            Brain* br = new Brain();
            br->getBrainProperties(brain);
            if(br->slices.size() > 0)
                taskmgr.addProcess(br);
        }
    }
    catch(...) {
        return 1;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    if(argc < 2) {
        cerr << "Invalid arguments" << endl;
        return 1;
    }
    if(!strcmp(argv[1], "-f")){
        if(argc == 5 || argc == 6) {
            /*
            Slice sl(argv[2]);
            sl.pixelSize = 0.4875;
            if(!sl.initiallized) {
                cerr << "Fail to initiallize slice data." << endl;
                return 1;
            }
            Stack op(1, argv[3], ".tif");
            double cutPoint = 0;
            if(argc == 6) cutPoint = atof(argv[5]);
            sl.genVoxelData(atof(argv[4]), op,
                            Point2d(sl.bound.xs + 50, sl.bound.xe - 50),
                            Point2d(sl.bound.ys + 50, sl.bound.ye - 50),
                            Point2d(cutPoint, sl.stacks[0].images[0].data.rows * sl.pixelSize / sqrt(2)));

            //debug output
            cout << "Saving" << endl;

            op.save();
            */
            TaskManager fm;
            Brain br = Brain(string("default"));
            Slice sl(argv[2], 0, 0.4875);
            sl.snapshotPath = string(argv[3]);
            br.slices.push_back(sl);
            GenVoxelTask v;
            v.scale = atof(argv[4]);
            fm.tasks.push_back(&v);
            fm.addProcess(&br);
            fm.run();
            return 0;
        }
    }
    if(!strcmp(argv[1], "-v")){
        if(argc >= 4){
            Slice sl(argv[2], 0, 0.4875);
            if(!sl.initiallized) {
                cerr << "Fail to initiallize slice data." << endl;
                return 1;
            }
            ofstream fs;
            for(size_t i = 0; i < sl.stacks.size(); i++){
                cout << string(argv[3]) + to_string(i) + string(".txt") <<endl;
                fs.open(string(argv[3]) + to_string(i) + string(".txt"));
                if(!fs.is_open()) cerr << "not opened" << endl;
                for(size_t j = 0; j < sl.stacks[i].images.size(); j++) {
                    fs << sl.stacks[i].images[j].posX << endl;
                }
                fs.close();                
            }

            if(argc >= 5) {
                int ct = 0;
                double ss = sl.bound.ys;
                double scale = atof(argv[4]);
                vector<int> splitL, splitR, snapshotIdx;
                double pixelScale =  scale / sl.pixelSize;
                for(uint i = 0; i < sl.stacks.size(); i++){
                    splitL.push_back(max(int((ss - sl.stacks[i].bound.ys) * pixelScale) - 1, 0));
                    splitR.push_back(int(floor((sl.stacks[i].bound.ye - ss) * pixelScale)) + splitL[i]);
                    snapshotIdx.push_back(ct);
                    ct += splitR[i] - splitL[i] - 1;
                    ss += (splitR[i] - splitL[i]) / pixelScale;
                }
                string fn = string(argv[3]) + string("dp.txt");
                cout << fn << endl;
                fs.open(fn);
                if(!fs.is_open()) cerr << "not opened" << endl;
                for(size_t j = 0; j < snapshotIdx.size(); j++) {
                    fs << snapshotIdx[j] << endl;
                }
                fs.close();
#pragma omp parallel for
                for(int i = 0; i < sl.stacks.size(); i++){
                    Stack sav;
                    sav.prefix = string(argv[3]) + to_string(i) + "_";
                    auto st = sl.stacks[i].genVoxelData(scale,
                                                        Point2d(sl.bound.xs, sl.bound.xe),
                                                        Point2d(sl.stacks[i].bound.ys, sl.stacks[i].bound.ye),
                                                        Point2d(0, sl.stacks[0].images[0].data.rows * sl.pixelSize / sqrt(2)),
                            sl.pixelSize);
                    for(auto im : st)
                        sav.append(im);
                    sav.save();
                    sav.release();
                }
            }
            return 0;
        }
    }
    if(!strcmp(argv[1], "-b")){
        if(argc >= 3){
            TaskManager tm;
            if(resolveInputFile(string(argv[2]), tm)) {
                cout << "Invalid inputfile" << endl;
                return 1;
            }
            tm.run();
            return 0;
        }
    }
    if(!strcmp(argv[1], "-s")){
        if(argc >= 5) {
            string srcFolder = string(argv[2]);
            if(srcFolder.back() == '\\' || srcFolder.back() == '/') srcFolder.pop_back();
            string srcName = srcFolder.substr(srcFolder.find_last_of("/\\") + 1, srcFolder.length());
            string snapFolder = string(argv[3]);
            if(snapFolder.back() == '\\' || snapFolder.back() == '/') snapFolder.pop_back();
#ifdef _WIN32
            string dir("dir /b /ad ");
            string del("del ");
            string mkdir("md ");
            string cat(" >");
            string tmpFile("fl.tmp");
#else
            string dir("ls ");
            string del("rm ");
            string mkdir("mkdir -p ");
            string cat(" > ");
            string tmpFile("fl.tmp");
#endif
            string ccmd = dir + quot(srcFolder) + cat + quot(tmpFile);
            if(system(ccmd.c_str())) return 1;
            fstream folderList;
            folderList.open(tmpFile);
            for(int i = 0; i < 10; i++) {
                if(folderList.is_open()) break;
                testRun(100, 101);
                folderList.open(tmpFile);
            }
            vector<string> folders;
            while(!folderList.eof()) {
                folders.push_back(string());
                folderList >> folders.back();
            }
            folders.pop_back();
            folderList.close();
            ccmd = del + quot(tmpFile);
            system(ccmd.c_str());

            TaskManager fm;
            Brain br = Brain(srcName);
#pragma omp parallel for
            for(int i = 0; i < folders.size(); i++) {
                string s = folders[i];
                if(s.back() == '/')
                    s.pop_back();
                Slice sl(srcFolder + "/" + s + "/" + s + ".flsm", i, 0.4875);
                if(!sl.initiallized) continue;
                if(sl.brainName.empty()) sl.brainName = "unknown";
                sl.snapshotPath = snapFolder + "/" + sl.brainName + "/"
                        + to_string(sl.idxZ) + "-" + sl.name + "-" + sl.captureTime + "/";
#ifdef _WIN32
                for(char& ch : sl.snapshotPath) {
                    if(ch == '/') ch = '\\';
                }
#endif
                ccmd = mkdir + quot(sl.snapshotPath);
#pragma omp critical
                {
                    system(ccmd.c_str());
                    br.slices.push_back(sl);
                }
            }

            if(br.slices.size() == 0) return 1;
            if(!strcmp(argv[4], "c")) {
                CellCountingTask v;
                v.saveDir = snapFolder;
                fm.tasks.push_back(&v);
                GenVoxelTask gvt;
                gvt.scale =0.125;
                fm.tasks.push_back(&gvt);
                fm.addProcess(&br);
                fm.run();
                return 0;
            }

            if(!strcmp(argv[4], "cc")) {
                CellCountingTask v;
                v.saveDir = snapFolder;
                fm.tasks.push_back(&v);
                fm.addProcess(&br);
                fm.run();
                return 0;
            }

            GenVoxelTask v;
            v.scale = atof(argv[4]);
            fm.tasks.push_back(&v);
            fm.addProcess(&br);
            fm.run();

            return 0;
        }
    }
    cerr << "Invalid arguments" << endl;
    return 0;
}
