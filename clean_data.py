import os

#simply remove first line indicating field values, then normalize all numbers from 0 to 1
def clean_data(csv_file_in, x_out, y_out):
    out_x = ""
    out_y = ""
    with open(csv_file_in, "r") as f:
        for line in f.read().split("\n")[1:]:
            GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research, Chance_of_Admit = line.split(",")[1:]
            GRE_Score = float(GRE_Score)/340.0
            TOEFL_Score = float(TOEFL_Score)/120.0
            University_Rating = float(University_Rating)/5.0
            SOP = float(SOP)/5.0
            LOR = float(LOR)/5.0
            CGPA = float(CGPA)/10.0
            out_x += str(GRE_Score) + ","
            out_x += str(TOEFL_Score) + ","
            out_x += str(University_Rating) + ","
            out_x += str(SOP) + ","
            out_x += str(LOR) + ","
            out_x += str(CGPA) + ","
            out_x += str(Research) + "\n"
            out_y += str(Chance_of_Admit) + "\n"
    with open(x_out, "w+") as f:
        f.write(out_x[:-1])
    with open(y_out, "w+") as f:
        f.write(out_y[:-1])


pwd = os.path.realpath(os.path.dirname(__file__))
in_csv = pwd + '/data/original/Past_Students.csv'
out_x_csv = pwd + '/data/clean/data_x.csv'
out_y_csv = pwd + '/data/clean/data_y.csv'

if __name__ == '__main__':
    clean_data(in_csv, out_x_csv, out_y_csv)
