#include "fslice.h"
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <thread>
#include <future>
#include <fstream>
#include <regex>
#include <list>
#include <iostream>

using namespace cv;
using namespace std;
using namespace flsm;

const double defalut_spacing = 5.005;
const double stageOffset = -60;

Slice::Slice() //test
{
    stacks.assign(12, Stack(1500));
}

Slice::Slice(string idxFileName, int idx_z, double pixSize, bool useVideo, box roi)
{
    idxFile = idxFileName;
    idxZ = idx_z;
    pixelSize = pixSize;
    string folderName = idxFile.substr(0, idxFile.find_last_of("/\\"));
    captureTime = folderName.substr(folderName.find_last_of("/\\") + 1, folderName.find_last_of("."));
    folderName = folderName + "/";
    ifstream jf(idxFileName);
    try {
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
        rapidjson::ParseResult pres = doc.Parse(buf);
        if(pres.IsError()) {
            cout << idxFileName << ": " << rapidjson::GetParseError_En(pres.Code()) << endl;
            throw 2;
        }
        if(!doc.IsObject()) throw 1;

        int slice_index = -1, slide_index = -1;
        if(doc.HasMember("slices_index")) {
            slice_index = stoi(doc["slices_index"].GetString());
        }
        else  {
            if(doc.HasMember("slice")) {
                string sli = doc["slice"].GetString();
                regex cap("\\d+_\\d+$");
                try {
                    if(regex_match(sli, cap)) {
                        slice_index = stoi(sli.substr(sli.find_last_of("_") + 1, sli.length()));
                        slide_index = stoi(sli.substr(0, sli.find_last_of("_")));
                    }
                }
                catch(regex_error& ) {

                }
            }
        }
        if(doc.HasMember("slides_index")) {
            slide_index = stoi(doc["slides_index"].GetString());
        }
        if(doc.HasMember("caption")) {
            name = doc["caption"].GetString();
            try {
                regex cap(".+_\\d+_\\d+$");
                if(regex_match(name, cap)) {
                    slice_index = stoi(name.substr(name.find_last_of("_") + 1, name.length()));
                    string ssli = name.substr(0, name.find_last_of("_"));
                    slide_index = stoi(ssli.substr(ssli.find_last_of("_") + 1, ssli.length()));
                    brainName = ssli.substr(0, ssli.find_last_of("_"));
                }
                else brainName = name;
            }
            catch(regex_error& ) {
                brainName = name;
            }
        }
        if(doc.HasMember("wave_length"))
            brainName = brainName + "_" + doc["wave_length"].GetString();
        if(doc.HasMember("power"))
            brightness = double(stoi(doc["power"].GetString())) / 10;
        int spacing = defalut_spacing;
        if(doc.HasMember("speed"))
            spacing = stod(doc["speed"].GetString()) * 10.01;
        if(slice_index != -1 && slide_index != -1)
            idxZ = slice_index * 8 + slide_index - 8;

        if(!doc.HasMember("images")) throw 3;
        if(!doc["images"].HasMember("motor")) throw 4;
        if(!doc["images"]["motor"].HasMember("0_50")) throw 5;
        if(!doc["images"]["motor"]["0_50"].IsObject()) throw 6;

        int flipcode = 0;
        if(doc.HasMember("rotation"))
            if(doc["rotation"].IsString()) {
                string fl = doc["rotation"].GetString();
                if(fl.compare("180") == 0)
                    flipcode = -1;
            }
        rapidjson::Value& imgIdx = doc["images"]["motor"];
        rapidjson::Value& img = imgIdx["0_50"];
        bound.xs = 1000 * img["position_x"].GetDouble();
        bound.ys = 1000 * img["position_y"].GetDouble();
        //int imgH = 2048;
        //if(index["imageHeight"].toInt(0) > 0) imgH = index["imageHeight"].toInt();

        for(int i = 0; true; i++){
            if(!imgIdx.HasMember((to_string(i) + "_50").c_str())) break;
            rapidjson::Value& image = imgIdx[(to_string(i) + "_50").c_str()];
            Image im(folderName + image["filename"].GetString());
            im.load(true);
            if(!im.isLoaded()) throw 10;
            Rect roi_(0, roi.zs * im.data.rows, im.data.cols, roi.ze * im.data.rows);
            Stack st(true, roi_);
            st.bound.xs = 1000 * image["position_x"].GetDouble();
            st.bound.ys = 1000 * image["position_y"].GetDouble();
            st.endct = 1;

            //Correct stage position
            vector<double> p1;

            for(int j = 1; true; j++){
                if(!imgIdx.HasMember((to_string(i) + "_" + to_string(j)).c_str())) break;
                rapidjson::Value& image = imgIdx[(to_string(i) + "_" + to_string(j)).c_str()];
                Image fi(folderName + image["filename"].GetString(), Mat(), roi_);
                fi.flipcode = flipcode;
                fi.posX = 1000 * image["position_x"].GetDouble();
                fi.posY = 1000 * image["position_y"].GetDouble();
                //fi.imageHeight = imgH;

                //Correct stage position
                if(j % 10 == 0)
                    p1.push_back(fi.posX);

                st.append(fi);
            }
            if(!getFilesInFolder(folderName + "thumbnail").empty()) {
                string i_ = imgIdx[(to_string(i) + "_50").c_str()]["filename"].GetString();
                i_ = i_.substr(i_.find_last_of("/") + 1, i_.find_first_of('_') - i_.find_last_of("/") - 1);
                st.videoFile = folderName + "thumbnail/" + i_ + "-images.h265";
                fstream alias_f(folderName + "thumbnail/" + i_ + "-info.txt");
                if(alias_f.is_open()) {
                    st.videoAlias.assign(st.images.size(), -1);
                    int ct = 0;
                    while(!alias_f.eof() && ct < st.images.size()) {
                        try {
                            string ns;
                            alias_f >> ns;
                            int n = stoi(ns.substr(ns.find_last_of("_") + 1, ns.length())) - 1;
                            st.videoAlias[n] = ct++;
                        }
                        catch(const std::invalid_argument&) {
                            continue;
                        }
                    }
                    if(useVideo) st.fromVideo = true;
                }
            }
            else if(!getFilesInFolder(folderName + "compressed").empty()) {
                st.videoFile = folderName + "compressed/" + to_string(i) + ".mp4";
                if(useVideo) st.fromVideo = true;
            }

            if(st.images.size() < 51) throw 6;
            st.bound.xe = st.images[st.images.size() - 1].posX;
            st.spaceing = (st.bound.xe - st.bound.xs) / (st.images.size() - 1);
            if(st.getImageData(0).cols == 0) throw 7;
            st.bound.ye = st.bound.ys + st.getImageData(0).cols * pixSize;

            //Correct stage position
            double mi = st.bound.xe > st.bound.xs ? 1 : -1;
            for(size_t i = 0; i < p1.size(); i++)
                p1[i] -= st.bound.xs + spacing * i * mi * 10;
            list<double> p2;
            for(size_t i = 6; i < p1.size() - 5; i++){
                if(fabs(p1[i] - p1[i + 1]) < 1 || fabs(p1[i] - p1[i - 1]) < 1)
                    p2.push_back(p1[i]);
            }
            for(int i = 0; i < 3; i++){
                double avr = 0, ste = 0;
                list<double>::iterator ite;
                for(ite = p2.begin(); ite != p2.end(); ++ite){
                    avr += (*ite);
                }
                avr /= p2.size();
                for(ite = p2.begin(); ite != p2.end(); ++ite)
                    ste += ((*ite) - avr) * ((*ite) - avr);
                ste = sqrt(ste / (p2.size() - 1));
                for(ite = p2.begin(); ite != p2.end(); ){
                    if(fabs((*ite) - avr) > ste)
                        ite = p2.erase(ite);
                    else ++ite;
                }
            }
            double avr = 0;
            for(list<double>::iterator ite = p2.begin(); ite != p2.end(); ++ite){
                avr += (*ite);
            }
            avr /= p2.size();
            st.bound.xs += avr + mi * stageOffset;
            st.spaceing = mi * spacing;
            st.bound.xe = st.bound.xs + st.spaceing * (st.images.size() - 1);

            stacks.push_back(st);
        }
    }
    catch (int err) {
        cout << err << endl;
        return;
    }

    for(auto st : stacks) {
        if(min(st.bound.xs, st.bound.xe) < bound.xs)
            bound.xs = min(st.bound.xs, st.bound.xe);
        if(max(st.bound.xs, st.bound.xe) > bound.xe)
            bound.xe = max(st.bound.xs, st.bound.xe);
    }
    bound.ye = stacks[stacks.size() - 1].bound.ye;
    bound.zs = 0;
    bound.ze = stacks[0].images[0].data.rows * pixSize / sqrt(2);
    snapshotPath = folderName + "result/";
    initiallized = true;
}

void Slice::genVoxelData(double scale, Stack& dst, Point2d xRange, Point2d yRange, Point2d zRange)
{
    if(initiallized == false) return;
    double ys = fmax(yRange.x, stacks[0].bound.ys);
    zRange = Point2d(max(zRange.x, 0.0), min(zRange.y, stacks[0].getImageData(0).rows * pixelSize / sqrt(2)));
    int ct = 0;
    vector<future<vector<Mat> > > fut;
    vector<vector<Mat> > res(stacks.size());
    for(size_t i = 0; i < stacks.size(); i++) {
        if(yRange.y < stacks[i].bound.ys) return;
        double se = stacks[i].bound.ys + stacks[i].getImageData(0).cols * pixelSize;
        if(yRange.x > se) continue;

        //debug output
        cout << "stack " << i << endl;

        int nt = floor((fmin(se, yRange.y) - ys) / pixelSize * scale);
        vector<Mat> sub;

        /*stacks[i].genVoxelData_(sub, scale, xRange,
                                                 Point2d(ys + ct * pixelSize / scale, ys + nt * pixelSize / scale),
                                                 zRange,
                                                 pixelSize);*/
        fut.push_back(async(launch::async, &Stack::genVoxelData, stacks[i],
                            scale, xRange,
                            Point2d(ys + ct * pixelSize / scale, ys + nt * pixelSize / scale),
                            zRange,
                            pixelSize));
        //res.push_back(sub);
        ct = nt;
    }
    //for(auto th : threads) th->join();
    for(auto &fu : fut){
        auto dd = fu.get();
        for(auto i : dd){
            dst.append(i);
        }
    }
}
