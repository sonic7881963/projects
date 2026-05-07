
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level1/mian.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'd4300SSbLBAmqdaWSSlm/Ka', 'mian');
// 3.16小游戏/command_TypeScript/level1/mian.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var main = /** @class */ (function (_super) {
    __extends(main, _super);
    function main() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    main_1 = main;
    main.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    main.prototype.onEventStart = function () {
        cc.log("click");
        main_1.attack = true;
    };
    var main_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    main.attack = false;
    main.minion_attack = false;
    main.main_hp = 163;
    main.minion1_hp = 163;
    main.minion2_hp = 163;
    main.minion3_hp = 163;
    __decorate([
        property(cc.Label)
    ], main.prototype, "label", void 0);
    __decorate([
        property
    ], main.prototype, "text", void 0);
    main = main_1 = __decorate([
        ccclass
    ], main);
    return main;
}(cc.Component));
exports.default = main;

cc._RF.pop();
                    }
                    if (nodeEnv) {
                        __define(__module.exports, __require, __module);
                    }
                    else {
                        __quick_compile_project__.registerModuleFunc(__filename, function () {
                            __define(__module.exports, __require, __module);
                        });
                    }
                })();
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDFcXG1pYW4udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7OztBQUFBLG9CQUFvQjtBQUNwQix3RUFBd0U7QUFDeEUsbUJBQW1CO0FBQ25CLGtGQUFrRjtBQUNsRiw4QkFBOEI7QUFDOUIsa0ZBQWtGOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFFNUUsSUFBQSxLQUFzQixFQUFFLENBQUMsVUFBVSxFQUFsQyxPQUFPLGFBQUEsRUFBRSxRQUFRLGNBQWlCLENBQUM7QUFHMUM7SUFBa0Msd0JBQVk7SUFBOUM7UUFBQSxxRUE0QkM7UUF6QkcsV0FBSyxHQUFhLElBQUksQ0FBQztRQUd2QixVQUFJLEdBQVcsT0FBTyxDQUFDOztJQXNCM0IsQ0FBQzthQTVCb0IsSUFBSTtJQW9CckIsb0JBQUssR0FBTDtRQUNJLElBQUksQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLFlBQVksRUFBRSxJQUFJLENBQUMsWUFBWSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3hELENBQUM7SUFDRCwyQkFBWSxHQUFaO1FBQ0ksRUFBRSxDQUFDLEdBQUcsQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUNoQixNQUFJLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztJQUN2QixDQUFDOztJQWxCRCx3QkFBd0I7SUFFeEIsZUFBZTtJQUVGLFdBQU0sR0FBWSxLQUFLLENBQUM7SUFDeEIsa0JBQWEsR0FBWSxLQUFLLENBQUM7SUFDL0IsWUFBTyxHQUFXLEdBQUcsQ0FBQztJQUN0QixlQUFVLEdBQVcsR0FBRyxDQUFDO0lBQ3pCLGVBQVUsR0FBVyxHQUFHLENBQUM7SUFDekIsZUFBVSxHQUFXLEdBQUcsQ0FBQztJQWR0QztRQURDLFFBQVEsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDO3VDQUNJO0lBR3ZCO1FBREMsUUFBUTtzQ0FDYztJQU5OLElBQUk7UUFEeEIsT0FBTztPQUNhLElBQUksQ0E0QnhCO0lBQUQsV0FBQztDQTVCRCxBQTRCQyxDQTVCaUMsRUFBRSxDQUFDLFNBQVMsR0E0QjdDO2tCQTVCb0IsSUFBSSIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgbWFpbiBleHRlbmRzIGNjLkNvbXBvbmVudCB7XHJcblxyXG4gICAgQHByb3BlcnR5KGNjLkxhYmVsKVxyXG4gICAgbGFiZWw6IGNjLkxhYmVsID0gbnVsbDtcclxuXHJcbiAgICBAcHJvcGVydHlcclxuICAgIHRleHQ6IHN0cmluZyA9ICdoZWxsbyc7XHJcblxyXG4gICAgLy8gTElGRS1DWUNMRSBDQUxMQkFDS1M6XHJcbiAgICBcclxuICAgIC8vIG9uTG9hZCAoKSB7fVxyXG4gICBcclxuICAgcHVibGljIHN0YXRpYyBhdHRhY2s6IGJvb2xlYW4gPSBmYWxzZTtcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb25fYXR0YWNrOiBib29sZWFuID0gZmFsc2U7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWFpbl9ocDogbnVtYmVyID0gMTYzO1xyXG4gICBwdWJsaWMgc3RhdGljIG1pbmlvbjFfaHA6IG51bWJlciA9IDE2MztcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb24yX2hwOiBudW1iZXIgPSAxNjM7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uM19ocDogbnVtYmVyID0gMTYzO1xyXG4gICBcclxuXHJcbiAgICBzdGFydCAoKSB7XHJcbiAgICAgICAgdGhpcy5ub2RlLm9uKCd0b3VjaHN0YXJ0JywgdGhpcy5vbkV2ZW50U3RhcnQsIHRoaXMpO1xyXG4gICAgfVxyXG4gICAgb25FdmVudFN0YXJ0KCkge1xyXG4gICAgICAgIGNjLmxvZyhcImNsaWNrXCIpO1xyXG4gICAgICAgIG1haW4uYXR0YWNrID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG59XHJcbiJdfQ==