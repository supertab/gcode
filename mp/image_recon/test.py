import matplotlib.font_manager
fonts = matplotlib.font_manager.findSystemFonts()
l = []
for f in fonts:
  try:
      font =matplotlib.font_manager.FontProperties(fname=f)
      #print(font.get_family())
      l.append((f, font.get_name(), font.get_family(), font.get_weight()))
  except:
      pass

print(l)