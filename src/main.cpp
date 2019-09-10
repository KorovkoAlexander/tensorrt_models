#include <Openpose.h>


int main(){
//    cout <<"Hello" << endl;
//    {
//        Openpose model(
//                "/home/alex/Code/Openpose/Openpose.engine",
//                "0",
//                {"351", "365"},
//                1,
//                TYPE_FP32
//        );
//        string s;
//        cin >> s;
//    }
//    string s, k;
//    cin >> s;
//    Openpose model(
//            "/home/alex/Code/Openpose/Openpose.engine",
//            "0",
//            {"351", "365"},
//            1,
//            TYPE_FP32
//    );
//    cin >> k;
//

//    float* imgCPU    = NULL;
//    float* imgCUDA   = NULL;
//    int    imgWidth  = 0;
//    int    imgHeight = 0;
//
//    const char* filename = "/home/alex/Code/OpenposeTensorRT/2.jpg";
//    const char* outfilename = "/home/alex/Code/OpenposeTensorRT/2.jpg";
//
//    if(!loadImageRGB(filename, (float3**)&imgCPU, (float3**)&imgCUDA, &imgWidth, &imgHeight) )
//    {
//        printf("failed to load image '%s'\n", filename);
//        return 0;
//    }
//
//    if (model.Apply(imgCUDA, imgWidth, imgHeight)){
//        cout << "All right!" << endl;
//    }
//
//    outputLayer heatmaps = model.get_output(OUTPUT_PAF);
//
//    float* data = heatmaps.CPU;
//    int w = DIMS_W(heatmaps.dims);
//    int h = DIMS_H(heatmaps.dims);
//    int c = DIMS_C(heatmaps.dims);
//
//    ofstream s("/home/alex/Code/OpenposeTensorRT/heatmaps.p");
//    for(int i = 0; i < heatmaps.size/4; i++){
//        s << data[i] << endl;
//    }

    return 0;
}