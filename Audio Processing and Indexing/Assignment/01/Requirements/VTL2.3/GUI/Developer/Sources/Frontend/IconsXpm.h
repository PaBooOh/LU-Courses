// ****************************************************************************
// This file is part of VocalTractLab.
// Copyright (C) 2020, Peter Birkholz, Dresden, Germany
// www.vocaltractlab.de
// author: Peter Birkholz
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//
// ****************************************************************************

#ifndef __ICONS_H__
#define __ICONS_H__


// ******************************************************************
// This is the icon for the program shown at the left end of the
// main window title bar.
// This xpm-structure was converted from a gif file using the 
// online converter under http://www.convertmyimage.com/index.php.
// ******************************************************************

static const char *xpmLogo16[] = {
/* columns rows colors chars-per-pixel */
"16 16 256 2",
"   c None",
".  c #000033",
"X  c #000066",
"o  c #000099",
"O  c #0000CC",
"+  c blue",
"@  c #003300",
"#  c #003333",
"$  c #003366",
"%  c #003399",
"&  c #0033CC",
"*  c #0033FF",
"=  c #006600",
"-  c #006633",
";  c #006666",
":  c #006699",
">  c #0066CC",
",  c #0066FF",
"<  c #009900",
"1  c #009933",
"2  c #009966",
"3  c #009999",
"4  c #0099CC",
"5  c #0099FF",
"6  c #00CC00",
"7  c #00CC33",
"8  c #00CC66",
"9  c #00CC99",
"0  c #00CCCC",
"q  c #00CCFF",
"w  c green",
"e  c #00FF33",
"r  c #00FF66",
"t  c #00FF99",
"y  c #00FFCC",
"u  c cyan",
"i  c #330000",
"p  c #330033",
"a  c #330066",
"s  c #330099",
"d  c #3300CC",
"f  c #3300FF",
"g  c #333300",
"h  c gray20",
"j  c #333366",
"k  c #333399",
"l  c #3333CC",
"z  c #3333FF",
"x  c #336600",
"c  c #336633",
"v  c #336666",
"b  c #336699",
"n  c #3366CC",
"m  c #3366FF",
"M  c #339900",
"N  c #339933",
"B  c #339966",
"V  c #339999",
"C  c #3399CC",
"Z  c #3399FF",
"A  c #33CC00",
"S  c #33CC33",
"D  c #33CC66",
"F  c #33CC99",
"G  c #33CCCC",
"H  c #33CCFF",
"J  c #33FF00",
"K  c #33FF33",
"L  c #33FF66",
"P  c #33FF99",
"I  c #33FFCC",
"U  c #33FFFF",
"Y  c #660000",
"T  c #660033",
"R  c #660066",
"E  c #660099",
"W  c #6600CC",
"Q  c #6600FF",
"!  c #663300",
"~  c #663333",
"^  c #663366",
"/  c #663399",
"(  c #6633CC",
")  c #6633FF",
"_  c #666600",
"`  c #666633",
"'  c gray40",
"]  c #666699",
"[  c #6666CC",
"{  c #6666FF",
"}  c #669900",
"|  c #669933",
" . c #669966",
".. c #669999",
"X. c #6699CC",
"o. c #6699FF",
"O. c #66CC00",
"+. c #66CC33",
"@. c #66CC66",
"#. c #66CC99",
"$. c #66CCCC",
"%. c #66CCFF",
"&. c #66FF00",
"*. c #66FF33",
"=. c #66FF66",
"-. c #66FF99",
";. c #66FFCC",
":. c #66FFFF",
">. c #990000",
",. c #990033",
"<. c #990066",
"1. c #990099",
"2. c #9900CC",
"3. c #9900FF",
"4. c #993300",
"5. c #993333",
"6. c #993366",
"7. c #993399",
"8. c #9933CC",
"9. c #9933FF",
"0. c #996600",
"q. c #996633",
"w. c #996666",
"e. c #996699",
"r. c #9966CC",
"t. c #9966FF",
"y. c #999900",
"u. c #999933",
"i. c #999966",
"p. c gray60",
"a. c #9999CC",
"s. c #9999FF",
"d. c #99CC00",
"f. c #99CC33",
"g. c #99CC66",
"h. c #99CC99",
"j. c #99CCCC",
"k. c #99CCFF",
"l. c #99FF00",
"z. c #99FF33",
"x. c #99FF66",
"c. c #99FF99",
"v. c #99FFCC",
"b. c #99FFFF",
"n. c #CC0000",
"m. c #CC0033",
"M. c #CC0066",
"N. c #CC0099",
"B. c #CC00CC",
"V. c #CC00FF",
"C. c #CC3300",
"Z. c #CC3333",
"A. c #CC3366",
"S. c #CC3399",
"D. c #CC33CC",
"F. c #CC33FF",
"G. c #CC6600",
"H. c #CC6633",
"J. c #CC6666",
"K. c #CC6699",
"L. c #CC66CC",
"P. c #CC66FF",
"I. c #CC9900",
"U. c #CC9933",
"Y. c #CC9966",
"T. c #CC9999",
"R. c #CC99CC",
"E. c #CC99FF",
"W. c #CCCC00",
"Q. c #CCCC33",
"!. c #CCCC66",
"~. c #CCCC99",
"^. c gray80",
"/. c #CCCCFF",
"(. c #CCFF00",
"). c #CCFF33",
"_. c #CCFF66",
"`. c #CCFF99",
"'. c #CCFFCC",
"]. c #CCFFFF",
"[. c red",
"{. c #FF0033",
"}. c #FF0066",
"|. c #FF0099",
" X c #FF00CC",
".X c magenta",
"XX c #FF3300",
"oX c #FF3333",
"OX c #FF3366",
"+X c #FF3399",
"@X c #FF33CC",
"#X c #FF33FF",
"$X c #FF6600",
"%X c #FF6633",
"&X c #FF6666",
"*X c #FF6699",
"=X c #FF66CC",
"-X c #FF66FF",
";X c #FF9900",
":X c #FF9933",
">X c #FF9966",
",X c #FF9999",
"<X c #FF99CC",
"1X c #FF99FF",
"2X c #FFCC00",
"3X c #FFCC33",
"4X c #FFCC66",
"5X c #FFCC99",
"6X c #FFCCCC",
"7X c #FFCCFF",
"8X c yellow",
"9X c #FFFF33",
"0X c #FFFF66",
"qX c #FFFF99",
"wX c #FFFFCC",
"eX c gray100",
"rX c black",
"tX c gray5",
"yX c gray10",
"uX c #282828",
"iX c #353535",
"pX c #434343",
"aX c #505050",
"sX c #5D5D5D",
"dX c gray42",
"fX c gray47",
"gX c #868686",
"hX c #939393",
"jX c gray63",
"kX c #AEAEAE",
"lX c #BBBBBB",
"zX c gray79",
"xX c gray84",
"cX c #E4E4E4",
"vX c #F1F1F1",
"bX c gray100",
"nX c black",
"mX c black",
"MX c black",
"NX c black",
"BX c black",
"VX c black",
"CX c black",
"ZX c black",
"AX c black",
"SX c black",
"DX c black",
"FX c black",
"GX c black",
"HX c black",
"JX c black",
"KX c black",
"LX c black",
"PX c black",
"IX c black",
"UX c black",
/* pixels */
"zXzXzXzXzXzXzXkXgXtXvXvXvXvXvXvX",
"zXzXzXzXzXzXzXkXtXgXtXvXvXvXvXvX",
"zXzXzXzXzXkXtXtXtXkXgXtXvXvXvXvX",
"zXzXzXzXzXzXzXzXzXzXkXgXtXvXvXvX",
"zXzXkXkXkXkXkXkXkXkXkXkXgXtXvXvX",
"zXkXgXgXgXgXgXgXgXgXkXkXkXgXtXvX",
"kXgXtXtXtXtXtXtXtXtXgXgXtXtXtXvX",
"kXtXvXvXvXvXvXvXvXvXtXgXtXvXvXvX",
"gXtXvXvXtXtXtXtXtXvXvXtXtXvXvXvX",
"gXtXvXtX2X2X2X2X2XtXvXvXvXvXvXvX",
"gXtXvXtX2X2X2X2X2X2XtXtXtXvXvXvX",
"gXtXvXtX2X2X2X2X2X2XtXgXtXvXvXvX",
"gXtXvXtX2X2X2X2X2X2XtXgXtXvXvXvX",
"gXtXvXvXtXtXtXtXtXtXtXtXtXvXvXvX",
"gXtXvXvXgXtXvXvXvXvXvXvXvXvXvXvX",
"gXtXvXvXgXtXvXvXvXvXvXvXvXvXvXvX"
};

// ****************************************************************************
// Windows expects icons for the toolbar to have the size 16 x 15 pixels !!
// ****************************************************************************

static const char *xpmRecord[] =
{
"16 15 4 1",      // Columns, rows, colors, chars-per-pixel
"  c None",
"+ c #ff0000",
". c #d00000",
"o c #000000",
"                ",
"      oooo       ",
"    oo++++oo     ",
"   o++++++++o    ",
"  o++++++++++o   ",
"  o++++++++++o   ",
" o++++++++++++o  ",
" o++++++++++++o  ",
" o++++++++++++o  ",
" o++++++++++++o  ",
"  o++++++++++o   ",
"  o++++++++++o   ",
"   o++++++++o    ",
"    oo++++oo     ",
"      oooo       ",
};

// ****************************************************************************
// ****************************************************************************

static const char *xpmPlayAll[] =
{
"16 15 4 1",      // Columns, rows, colors, chars-per-pixel
"  c None",
"+ c #00d000",
". c #008000",
"o c #000000",
"                ",
"  ooo           ",
"  o++oo         ",
"  o++++oo       ",
"  o++++++oo     ",
"  o++++++++oo   ",
"  o++++++++++oo ",
"  o++++++++++++o",
"  o++++++++++oo ",
"  o++++++++oo   ",
"  o++++++oo     ",
"  o++++oo       ",
"  o++oo         ",
"  ooo           ",
"                ",
};

// ****************************************************************************
// ****************************************************************************

static const char *xpmPlayPart[] =
{
"16 15 4 1",      // Columns, rows, colors, chars-per-pixel
"  c None",
"+ c #00d000",
". c #008000",
"o c #000000",
"                ",
"o              o",
"o  oo          o",
"o  o+oo        o",
"o  o+++oo      o",
"o  o+++++oo    o",
"o  o+++++++oo  o",
"o  o+++++++++o o",
"o  o+++++++oo  o",
"o  o+++++oo    o",
"o  o+++oo      o",
"o  o+oo        o",
"o  oo          o",
"o              o",
"                ",
};

// ****************************************************************************
// ****************************************************************************

static const char *xpmClear[] =
{
"16 15 4 1",      // Columns, rows, colors, chars-per-pixel
"  c None",
"+ c #000000",
". c #202020",
"o c #000000",
"                ",
"                ",
" ++         ++  ",
"  ++       ++   ",
"   ++     ++    ",
"    ++   ++     ",
"     ++ ++      ",
"      +++       ",
"      +++       ",
"     ++ ++      ",
"    ++   ++     ",
"   ++     ++    ",
"  ++       ++   ",
" ++         ++  ",
"                ",
};

// ****************************************************************************
// ****************************************************************************

#endif
