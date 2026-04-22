#pragma once

#include <iostream>
#include <string>
#include <helper_string.h>

class CmdArgParser
{
public:
    CmdArgParser(int argc, char **argv)
    {
        char *in_path = nullptr;
        if (checkCmdLineFlag(argc, (const char **)argv, "input"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &in_path);
            input_file = in_path;
        }
        else
        {
            input_file = "data/Lena.png";
        }

        char *out_path = nullptr;
        if (checkCmdLineFlag(argc, (const char **)argv, "output"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "output", &out_path);
            output_file = out_path;
        }
        else
        {
            output_file = "data/Lena_edges.png";
        }

        file_extension = extractExtension(input_file);
    }

    std::string getInputFile() const { return input_file; }
    std::string getOutputFile() const { return output_file; }
    std::string getExtension() const { return file_extension; }

private:
    std::string input_file;
    std::string output_file;
    std::string file_extension;

    static std::string extractExtension(const std::string &path)
    {
        size_t pos = path.find_last_of('.');
        if (pos != std::string::npos)
        {
            std::string ext = path.substr(pos);
            for (auto &c : ext)
                c = tolower(c);
            return ext;
        }
        return "";
    }
};
