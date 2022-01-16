import pandas as pd
import sys,os
from docxtpl import DocxTemplate

i=1
def process(args):
    print('输入参数:公司信息表.xlsx 模板.docx 输出路径:C:\out\\')
    print('input args:{}'.format(args[1:]))
    company_info=pd.read_excel(r'{}'.format(args[1]), converters = {'期末余额': str})
    tpl=DocxTemplate(r'{}'.format(args[2]))
    out_path_tmp=r'{}'.format(args[3])

    #company_info=pd.read_excel(r"C:\Users\kexin\Desktop\test\company.xlsx", converters = {'期末余额': str})
    #tpl=DocxTemplate(r"C:\Users\kexin\Desktop\test\template.docx")
    #out_file_tmp= r'C:\Users\kexin\Desktop\test\out\{}_{}.docx'

    if not os.path.exists(out_path_tmp):
        print("create output path:", out_path_tmp)
        os.mkdir(out_path_tmp)

    def process_func(row):
        global i
        company = row['单位名称'].replace("*", '').replace('\\', '')
        try:
            #if not company.endswith('公司'):
            #company+='公司'
            pay=row['期末余额']

            out_file = os.path.join(out_path_tmp, '{}_{}.docx'.format(i, company))
            context = {
                'company' : company,
                'pay': '{:,.2f}'.format(float(pay)), # 逗号是千分位分隔符
            }
            tpl.render(context)
            tpl.save(out_file)
            print("{} done, output file:{}".format(i,  out_file))
        except Exception as e:
            print("process company {} error, detail:{}", company, e)
        i+=1

    company_info.apply(process_func, axis=1)
    print("kexin remind you: all job done!")

if __name__ == '__main__':
    process(sys.argv)
