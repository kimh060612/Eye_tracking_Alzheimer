import os
import xml.etree.ElementTree as ET

class_ = ["credit_card"]

file_path = "./Credit_Card_Plus_renew/"
file_list = os.listdir(file_path)
save_file = "./annotation.txt"
xml_file_list = []

F = open(save_file, "w")

for con in file_list:
    if con.split(".")[1] == "xml":
        xml_file_list.append(con)

for con in xml_file_list:
    try:
        total_text_line = ""

        xml_path = "./Credit_Card_Plus_renew/" + con
        Tree = ET.parse(xml_path)
        root = Tree.getroot()

        file_name = root.find("filename").text

        if not file_name in file_list:
            continue

        total_text_line += file_path + file_name

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in class_ or int(difficult)==1:
                continue
            cls_id = class_.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            total_text_line += " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
        
        total_text_line += "\n"

        F.write(total_text_line)
    except:
        print("Fail")

F.close()








