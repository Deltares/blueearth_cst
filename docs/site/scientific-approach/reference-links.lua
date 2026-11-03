-- Methodology citations point to the single shared bibliography after citeproc.
function Pandoc(doc)
  if quarto.doc.input_file:match("references%.qmd$") then
    return doc
  end
  -- Quarto resolves figure/table/section references later; keep them out of citeproc.
  local crossrefs = {}
  doc = doc:walk({Cite = function(cite)
    local id = cite.citations[1].id
    if id:match("^fig%-") or id:match("^tbl%-") or id:match("^sec%-") then
      crossrefs[#crossrefs + 1] = cite
      return pandoc.Span({}, pandoc.Attr("", {}, {["methodology-crossref"] = tostring(#crossrefs)}))
    end
  end})
  -- Generate linked citations before removing the local reference list.
  doc.meta["suppress-bibliography"] = false
  doc = pandoc.utils.citeproc(doc)
  doc.meta["suppress-bibliography"] = true
  doc = doc:walk({
    Div = function(div)
      if div.identifier == "refs" then return {} end
    end,
    Cite = function(cite)
      -- Preserve the formatted links through Quarto's later citeproc pass.
      return pandoc.Span(cite.content, pandoc.Attr("", {"citation"}))
    end
  })
  doc = doc:walk({Link = function(link)
    if link.target:match("^#ref%-") then
      link.target = "references.html" .. link.target
      return link
    end
  end})
  return doc:walk({Span = function(span)
    local index = span.attributes["methodology-crossref"]
    if index then return crossrefs[tonumber(index)] end
  end})
end
