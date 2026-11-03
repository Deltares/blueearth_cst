-- Apply preflighted publication links to authored site wrappers at render time.
-- Install in docs/site/_quarto.yml: filters: [../../dev/scripts/publication_links.lua]
-- Canonical sources are never rewritten. Quarto supplies the current input path:
-- https://quarto.org/docs/extensions/lua-api.html#document-metadata
local function normalize(path)
  return path:gsub("\\", "/")
end

function Pandoc(doc)
  if doc.meta["publication-siblings"] then
    -- Complete manuals embed their images, but not the files targeted by
    -- image-only links. Keep the embedded image without a broken file link.
    -- Website chapters retain their links to distributed full-size assets.
    doc = doc:walk({Link = function(link)
      local extension = link.target:lower():match("%.([%a]+)$")
      local local_image = not link.target:match("^[%a][%w+.-]*:")
          and not link.target:match("^//")
          and ({svg=true, png=true, jpg=true, jpeg=true, gif=true, webp=true})[extension]
      if #link.content == 1 and link.content[1].tag == "Image" and local_image then
        return link.content
      end
    end})
    local siblings = {}
    for _, name in ipairs(doc.meta["publication-siblings"]) do
      siblings[pandoc.utils.stringify(name)] = true
    end
    if quarto.doc.is_format("typst") then
      return doc:walk({Link = function(link)
        local volume, anchor = link.target:match("^([a-z][a-z0-9-]*)%.html#(.+)$")
        if siblings[volume] then
          link.target = volume .. ".pdf#nameddest=" .. anchor
        end
        return link
      end})
    end
    return doc
  end
  local input = normalize(quarto.doc.input_file)
  local root = pandoc.path.directory(input)
  -- Standalone/hidden-tree smoke renders have no Quarto project discovery.
    while root ~= pandoc.path.directory(root) do
      local probe = io.open(root .. "/technical-manual/publication-links.json", "r")
      if probe then
        probe:close()
        break
      end
      root = pandoc.path.directory(root)
    end
  local handle = assert(io.open(root .. "/technical-manual/publication-links.json", "r"),
    "Missing publication link map for " .. input .. "; run stage_site_guides.py before rendering")
  local maps = quarto.json.decode(handle:read("*a"))
  handle:close()
  local relative = input:sub(#root + 2)
  local links = maps[relative]
  if not links then
    return doc
  end
  return doc:walk({Link = function(link)
    local target = links[link.target]
    if target then
      link.target = target
      if target:match("^https?://.*github.com/.*/blob/") then
        link.content:insert(pandoc.Str(" (repository/online reference)"))
      end
    end
    return link
  end})
end
