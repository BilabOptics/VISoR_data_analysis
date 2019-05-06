#include "fbrain.h"

#include <iostream>
#include <fstream>
#include <regex>

using namespace cv;
using namespace std;
using namespace flsm;

Point3d flsm::convertPos(Point2d pos2d, pointer p)
{
    double pixSize = p.slice()->pixelSize;
    double x = -pos2d.y * pixSize / sqrt(2) + p.stack()->bound.xs + p.stack()->spaceing * p.im;
    double y = pos2d.x * pixSize + p.stack()->bound.ys;
    double z = pos2d.y * pixSize / sqrt(2);
    return cv::Point3d(x, y, z);
}

void flsm::inverseConvertPos(Point3d pos3d, pointer& p, Point2d& pos2d)
{
    double pixSize = p.slice()->pixelSize;
    pos2d.x = (pos3d.y - p.stack()->bound.ys) / pixSize;
    pos2d.y = pos3d.z / pixSize * sqrt(2);
    p.im = ((pos3d.x + pos3d.z) - p.stack()->bound.xs) / p.stack()->spaceing;
}

Brain::Brain(string Name)
{
    name = Name;
}

void flsm::Brain::getBrainProperties(rapidjson::Value &v)
{
    double pixelSize = 0.4875;
    double brightness = 1;
    string saveDir;
    if(v.HasMember("name"))
        if(v["name"].IsString()) name = v["name"].GetString();
    if(v.HasMember("pixelSize"))
        if(v["pixelSize"].IsDouble()) pixelSize = v["pixelSize"].GetDouble();
    if(v.HasMember("brightness"))
        if(v["brightness"].IsDouble()) brightness = v["brightness"].GetDouble();
    if(v.HasMember("saveDir"))
        if(v["saveDir"].IsString()) {
            saveDir = v["saveDir"].GetString();
            if(saveDir[saveDir.size() - 1] != '/') saveDir.push_back('/');
        }
    bool use_filter = false;
    regex filter;
    if(v.HasMember("nameFilter")) {
        try {
            filter = regex(v["nameFilter"].GetString());
            use_filter = true;
        }
        catch(regex_error&) {
            cerr << "Invalid regular expression, filter disabled." << endl;
        }
    }
    box rct;
    if(v.HasMember("zStart"))
        if(v["zStart"].IsDouble()) rct.zs = v["zStart"].GetDouble();
    if(v.HasMember("zEnd"))
        if(v["zEnd"].IsDouble()) rct.ze = v["zEnd"].GetDouble();
    if(v.HasMember("slices")) {
        if(v["slices"].IsObject()) {
//#pragma omp parallel for
            for(uint j = 0; j < v["slices"].MemberCount(); j++) {
                rapidjson::Value& slice = v["slices"][to_string(j).c_str()];
                if(!slice.HasMember("index")) continue;
                box rct_ = rct;
                if(slice.HasMember("zStart"))
                    if(slice["zStart"].IsDouble()) rct_.zs = slice["zStart"].GetDouble();
                if(slice.HasMember("zEnd"))
                    if(slice["zEnd"].IsDouble()) rct_.ze = slice["zEnd"].GetDouble();
                Slice sl(slice["index"].GetString(), j, pixelSize, false, rct_);
                if(!sl.initiallized) {
                    cout << slice["index"].GetString() << endl;
                    continue;
                }
                sl.brightness = brightness;
                if(slice.HasMember("snapshotPath"))
                    if(slice["snapshotPath"].IsString()) {
                        sl.snapshotPath = string(slice["snapshotPath"].GetString());
                        createFolder(sl.snapshotPath);
                    }
                slices.push_back(sl);
            }
        }
    }
    vector<string> folders;
    vector<int> idx_list;
    if(v.HasMember("directory")) {
        if(!v["directory"].IsString()) return;
        string path = v["directory"].GetString();
        if(path[path.size() - 1] != '/') path.push_back('/');
        folders = getFoldersInFolder(path);
        for(string& folder : folders) {
            folder = path + folder;
        }
    }
    if(v.HasMember("sliceList")) {
        if(!v["sliceList"].IsString()) return;
        string path = v["sliceList"].GetString();
        ifstream slice_list;
        slice_list.open(path);
        while(!slice_list.fail()) {
            string folder, idx_cap;
            slice_list >> idx_cap >> folder;
            folders.push_back(folder);
            try {
                regex cap("\\d+_\\d+$");
                if(regex_match(idx_cap, cap)) {
                    int pos = stoi(idx_cap.substr(idx_cap.find_first_of("_") + 1, name.length()));
                    int sli = stoi(idx_cap.substr(0, idx_cap.find_first_of("_")));
                    idx_list.push_back(pos * 8 + sli - 8);
                }
                else idx_list.push_back(-1);
            }
            catch(regex_error& ) {
                idx_list.push_back(-1);
            }
        }
    }
#pragma omp parallel for
    for(int i = 0; i < folders.size(); i++) {
        string folder = folders[i];
        vector<string> files = getFilesInFolder(folder);
        for(auto file : files) {
            if(file.substr(file.find_last_of(".") + 1, file.length()).compare("flsm") == 0) {
                Slice sl(folder + "/" + file, -1, pixelSize, false, rct);
                if(!sl.initiallized) {
                    cout << "fail:" << folder << endl;
                    continue;
                }
                if (use_filter) {
                    cout << "skip:" << folder << endl;
                    if(!regex_match(sl.brainName, filter))
                        continue;
                }
                cout << "pass:" << folder << endl;
                if(idx_list.size() == folders.size()) sl.idxZ = idx_list[i];
                sl.brightness = brightness;
                sl.snapshotPath = saveDir + sl.brainName + "/" + to_string(sl.idxZ) + "-" + sl.name + "-" + sl.captureTime + "/";
#pragma omp critical(slice_init)
                {
                    createFolder(sl.snapshotPath);
                    slices.push_back(sl);
                }
            }
        }
    }
}

pointer::pointer(Brain *brain_, int slice_, int stack_, int image_)
{
    brain = brain_;
    sl = slice_;
    st = stack_;
    im = image_;
}

bool pointer::operator<(const pointer other) const
{
    if(other.brain == nullptr) return false;
    if(brain == nullptr) return true;
    if(brain == other.brain){
        if(sl == other.sl){
            if(st == other.st){
                if(im < other.im ) return true;
                else return false;
            }
            else if(st < other.st) return true;
        }
        else if(sl < other.sl) return true;
    }
    else if(brain < other.brain) return true;
    return false;
}

Slice *pointer::slice() const
{
    return &(brain->slices[sl]);
}

Stack *pointer::stack() const
{
    return &(brain->slices[sl].stacks[st]);
}

Image *pointer::image() const
{
    return &(brain->slices[sl].stacks[st].images[im]);
}

int pointer::numSlice() const
{
    return brain->slices.size();
}

int pointer::numStack() const
{
    return brain->slices[sl].stacks.size();
}

int pointer::numImage() const
{
    return brain->slices[sl].stacks[st].images.size();
}

void pointer::print()
{
    cout << "brain\t" << brain->name;
    if(sl == -1)
        cout << "\tbegin" << endl;
    else if(sl == numSlice())
        cout << "\tend" << endl;
    else {
        cout << "\tslice\t" << slice()->idxZ;
        if(st == -1)
            cout << "\tbegin" << endl;
        else if(st == numStack())
            cout << "\tend" << endl;
        else {
            cout << "\tstack\t" << st;
            if(im == -1)
                cout << "\tbegin" << endl;
            else if(im == numImage())
                cout << "\tend" << endl;
            else cout << "\timage\t" << im << endl;
        }
    }
}

bool pointer::isValid(bool strict)
{
    int t;
    if(strict) t = -1;
    else t = 0;
    if(sl >= numSlice()) return false;
    if(sl < t) return false;
    if(st >= numStack()) return false;
    if(st < t) return false;
    if(sl >= numImage()) return false;
    if(sl < t) return false;
    return true;
}

void pointer::setValid(bool strict)
{
    int t;
    if(strict) t = -1;
    else t = 0;
    if(sl > numSlice()) sl = numSlice() - 1;
    if(sl < t) sl = t;
    if(st > numStack()) st = numStack() - 1;
    if(st < t) st = t;
    if(im > numImage()) im = numImage() - 1;
    if(im < t) im = t;
}
