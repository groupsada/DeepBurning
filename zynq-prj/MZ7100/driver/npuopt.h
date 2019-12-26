#ifndef __NPUOPT_H__
#define __NPUOPT_H__

#ifdef __cplusplus
extern "C" {
#endif


/*
	用二进制表示的输入文件（文件内容全部是0和1组成）中，每个字节表示 1 bit，32字节表示一个实际的整形数据
	用十六进制表示的输入文件（文件内容全部是0~f/F组成）中，每个字节表示 4bit，8个字节表示一个实际的整形数据
*/
enum data_type
{
    DATA_TYPE_BIN = 32,
    DATA_TYPE_HEX = 8,
};


/* 支持的文件类型 */
enum file_type
{
    FILE_TYPE_INVALID = 0,
    FILE_TYPE_DATA,
    FILE_TYPE_WEIGHT,
    FILE_TYPE_INDEX,
    FILE_TYPE_BIAS,
    FILE_TYPE_LUT,
    FILE_TYPE_INST,
    FILE_TYPE_OUTPUT,
    FILE_TYPE_MAX = 8,
};


/* 文件的相关信息 */
typedef struct _file_info
{
    char              *file;           /* 文件名称，包含完整的路径 */
    enum data_type     dtype;          /* 文件数据类型，支持二进制和十六进制 */
    enum file_type     ftype;          /* 文件类型 */
    unsigned int       memaddr;        /* 文件在内存中的物理地址 */
    unsigned int       cmdaddr;        /* 编译器生成的地址 */
    unsigned int       *virtualaddr;   /* Linux应用层的虚拟地址*/
	unsigned int       lines_len;      /* 文件中每一行的长度（单位：字节），自动获取 */
	unsigned int       col_len;        /* 文件中每一列的长度（单位：字节），自动获取 */
} file_info;


#define INIT_FILE_INFO(file, dtype, ftype, memaddr, cmdaddr) \
{ file, dtype, ftype, memaddr,  cmdaddr }


#define ARRAY_SIZE(array) (sizeof(array)/sizeof(array[0]))


/*
 * 函数功能： NPU 运算初始化，并加载NPU编译器生成的权重文件；
 *            当调用libnpu中的函数时，必须首先调用该函数初始化，否则会导致段错误
 * 参数说明：
 *         filetab      权重文件信息表
 *         filenum      权重文件的数量
 *
 * 返 回 值： 0： 初始化成功
 *           -1：初始化失败
 */
int npu_init(file_info filetab[], int filenum);


/*
 * 函数功能： 释放运行内存
 * 参数说明：
 *         无
 *
 * 返 回 值：无
 */
void npu_uninit();


/*
 * 函数功能：使能一次 NPU 运算
 * 参数说明：
 *         inst_start   inst 指令在内存中的物理地址
 * 		   inst_depth   inst 指令深度，即：inst文件中，每一列的长度
 * 返 回 值： 0： 使能成功
 *           -1： 使能失败
 */
int nup_enable(unsigned int inst_start, unsigned int inst_depth);


/*
 * 函数功能：把文件加载到内存中的指定地址中
 * 参数说明：
 *         filename   文件路径
 * 		   addr       内存中的地址
 *         dtype      数据类型，目前仅支持 bin/hex 两种，bin：DATA_TYPE_BIN，hex：DATA_TYPE_HEX
 *         fileline   返回文件中，行数，如果不需要请赋值为 NULL
 *         filecol    返回文件中，列数，如果不需要请赋值为 NULL
 * 返 回 值：无
 */
int npu_write_file2ddr(const char *filename, unsigned int *addr, enum data_type dtype, unsigned int *fileline, unsigned int *filecol);

 
/*
 * 函数功能：调试功能，按整形（int 型）的格式，把内存中的数据保存到文件中
 * 参数说明：
 *         line   文件中，每一行的长度，即：int 型数据的个数
 * 		   col    文件中，每一列的长度，即：int 型数据的个数
 * 		   addr   内存中的地址
 * 		   path   保存文件的路径
 * 返 回 值：无
 */
void npu_write_date2file(unsigned int line, unsigned int col, unsigned int *addr, const char *path);


/*
 * 函数功能：调试功能，按整形（int 型）打印出内存中的数据
 * 参数说明：
 *         line   文件中，每一行的长度，即：int 型数据的个数
 * 		   col    文件中，每一列的长度，即：int 型数据的个数
 * 		   addr   所要保存数据的起始地址
 * 返 回 值：无
 */
void debug_func(unsigned int line_len, unsigned int col_len, unsigned int *addr);


/*
 * 函数功能：调试功能，把 NPU 输出结果与示例文件做比对
 * 参数说明：
 *         filename  示例文件名（包含完整的路径）
 *         addr      NUP执行结果的地址
 * 		   dtype     数据格式（二进制或者十六进制）
 * 返 回 值：无
 */
 
void output_cmp(const char *filename, unsigned int *addr, enum data_type dtype);

#ifdef __cplusplus
}  
#endif

#endif // __NPUOPT_H__
